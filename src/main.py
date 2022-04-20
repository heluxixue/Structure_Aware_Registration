
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import torch as th
import numpy as np
import SimpleITK as sitk
from dirlab.dirlabhelper import Dataloader
import torch.nn.functional as F
import copy
import airlab as al
from mymedicallib import utils as mut
import json
import pandas as pd
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def create_mask_pyramid(mask,image_pyramid):
    mask_pyramid = []
    image_dim = 3
    for img in image_pyramid:
        shape = img.image.shape[2:]
        mask_sample = F.interpolate(mask.image,size=shape,mode='trilinear')
        mask_sample[mask_sample>=0.5]=1
        mask_sample[mask_sample<0.5]=0
        mask_size = img.size[-image_dim:]
        mask_spacing = img.spacing
        mask_origin = img.origin
        mask_pyramid.append(al.Image(mask_sample, mask_size, mask_spacing, mask_origin))
    return mask_pyramid



def gap_overlap(vol1,vol2,label=1):
    vol1[vol1>0] = 1
    vol1[vol1<1] = 0
    vol2[vol2>0] = 1
    vol2[vol2<1] = 0
    vol1l = vol1 == label
    vol2l = vol2 == label
    s = np.sum(np.logical_and(vol1l,vol2l))
    vol1l = ~vol1l
    vol2l = ~vol2l
    o = np.sum(np.logical_and(vol1l,vol2l))
    return s,o

using_landmarks = True



class Registration():
    def __init__(self,configpath,dataconfigpath,appRoot):
        self.config = None
        with open(configpath) as f:
            self.config = json.load(f)
        self.dataconfigpath = dataconfigpath
        self.dtype = th.float32
        self.device = 'cuda:0'
        self.appRoot = appRoot

    def loaddata(self,case='1'):
        """
        load image
        """
        print("loading images")
        #loader = Dataloader(self.config['dataset']['dataset'], self.dataconfigpath,True,True)
        loader = Dataloader("dirlab", self.dataconfigpath,True,True)

        global_fix =  al.Image(loader.GetVolumeData(case=case,T='00',voltype='sitk'),self.dtype,self.device)
        global_mov =  al.Image(loader.GetVolumeData(case=case,T='50',voltype='sitk'),self.dtype,self.device)
        global_fix, global_mov = al.utils.normalize_images(global_fix, global_mov)

        lung_fix = loader.GetLungMask(case=case,T='00',voltype='sitk')
        lung_mov = loader.GetLungMask(case=case,T='50',voltype='sitk')

        bone_fix = loader.GetBoneMask(case=case,T='00',voltype='sitk')
        bone_mov = loader.GetBoneMask(case=case,T='50',voltype='sitk')


        body_fix = mut.getbody(lung_fix)
        body_mov = mut.getbody(lung_mov)

        surface_fix, spine_fix = mut.getlungsurface(lung_fix,bone_fix)
        other_fix = mut.getother(surface_fix,spine_fix)

        

        lung_fix = al.Image(lung_fix,self.dtype,self.device)
        lung_fix.image = lung_fix.image*global_fix.image
        lung_mov = al.Image(lung_mov,self.dtype,self.device)
        lung_mov.image = lung_mov.image*global_mov.image

        bone_fix = al.Image(bone_fix,self.dtype,self.device)
        bone_fix.image = bone_fix.image*global_fix.image
        bone_mov = al.Image(bone_mov,self.dtype,self.device)
        bone_mov.image = bone_mov.image*global_mov.image

        body_fix = al.Image(body_fix,self.dtype,self.device)
        body_fix.image = body_fix.image*global_fix.image
        body_mov = al.Image(body_mov,self.dtype,self.device)
        body_mov.image = body_mov.image*global_mov.image

        surface_fix = al.Image(surface_fix,self.dtype,self.device)
        spine_fix = al.Image(spine_fix,self.dtype,self.device)
        other_fix = al.Image(other_fix,self.dtype,self.device)

        points_fix = loader.GetPtsData(case=case,T='00',spacing=loader.GetVolSpacing(case))
        points_mov = loader.GetPtsData(case=case,T='50',spacing=loader.GetVolSpacing(case))
        points = {'fix':points_fix,'mov':points_mov}
        pyramid = self.config['hyperparameter']['pyramid']
        im_pyramid={}
        mask_pyramid = {}
        if self.config['method']['is_global_loss']:
            im_pyramid['global'] = {
                'mov':al.create_image_pyramid(global_mov, pyramid),
                'fix':al.create_image_pyramid(global_fix, pyramid)
            }
        if self.config['method']['is_bone_loss']:
            im_pyramid['bone'] = {
                'mov':al.create_image_pyramid(bone_mov, pyramid),
                'fix':al.create_image_pyramid(bone_fix, pyramid),
            }
        if self.config['method']['is_lung_loss']:
            im_pyramid['lung'] = {
                'mov':al.create_image_pyramid(lung_mov, pyramid),
                'fix':al.create_image_pyramid(lung_fix, pyramid),
            }
        if self.config['method']['is_body_loss']:
            im_pyramid['body'] = {
                'mov':al.create_image_pyramid(body_mov, pyramid),
                'fix':al.create_image_pyramid(body_fix, pyramid),
            }
            
        if self.config['method']['is_surface_reg']:
            mask_pyramid['surface'] = create_mask_pyramid(surface_fix, im_pyramid['lung']['fix'])
        if self.config['method']['is_spine_reg']:
            mask_pyramid['spine'] = create_mask_pyramid(spine_fix, im_pyramid['lung']['fix'])
        if self.config['method']['is_other_reg']:
            mask_pyramid['other'] =  create_mask_pyramid(other_fix, im_pyramid['lung']['fix'])
        size_pyramid = []
        for k in im_pyramid:
            for vol in im_pyramid[k]['mov']:
                size_pyramid.append(vol.size)
            if len(size_pyramid)>0:
                break
        return im_pyramid,mask_pyramid,points,size_pyramid

    def setLoss(self,im_pyramid,mask_pyramid,level):
        """
        set loss function
        """
        loss = []
        

        if self.config['method']['is_global_loss']:
            global_fix_level = im_pyramid['global']['fix'][level]
            global_mov_level = im_pyramid['global']['mov'][level]
            if self.config['method']['im_sim'] =='ncc':
                mloss = al.loss.pairwise.NCC(global_fix_level, global_mov_level)
            elif self.config['method']['im_sim'] =='ngf':
                mloss = al.loss.pairwise.NGF(global_fix_level, global_mov_level)
            elif self.config['method']['im_sim'] == 'mi':
                mloss = al.loss.pairwise.MI(global_fix_level, global_mov_level)
            mloss.set_loss_weight(1)
            loss.append(mloss)

        if self.config['method']['is_lung_loss']:
            lung_fix_level = im_pyramid['lung']['fix'][level]
            lung_mov_level = im_pyramid['lung']['mov'][level]
            mloss = al.loss.pairwise.NGF(lung_fix_level,lung_mov_level)                                 
            mloss.set_loss_weight(self.config['hyperparameter']['lung_w'])
            loss.append(mloss)

        if self.config['method']['is_bone_loss']:
            bone_fix_level = im_pyramid['bone']['fix'][level]
            bone_mov_level = im_pyramid['bone']['mov'][level]
            mloss = al.loss.pairwise.NCC(bone_fix_level,bone_mov_level)
            mloss.set_loss_weight(self.config['hyperparameter']['bone_w'])
            loss.append(mloss)

        if self.config['method']['is_body_loss']:
            body_fix_level = im_pyramid['body']['fix'][level]
            body_mov_level = im_pyramid['body']['mov'][level]
            mloss = al.loss.pairwise.NCC(body_fix_level,
                                            body_mov_level)
            mloss.set_loss_weight(self.config['hyperparameter']['body_w'])
            loss.append(mloss)
        return loss

    def setRegularization(self,im_pyramid,mask_pyramid,level):
        """
        set regularzation
        """
        reg = []
        key = None
        for k in im_pyramid:
            key = k
            break
        global_fix_level = im_pyramid[key]['fix'][level]
        if self.config['method']['is_global_reg']:
            if self.config['method']['is_pTV']:
                TVreg = al.regulariser.parameter.IsotropicTVRegulariser(global_fix_level.spacing)
                TVreg.set_weight(self.config['hyperparameter']['global_r_w'])
                reg.append(TVreg)
            elif self.config['method']['is_pSM']:
                diffreg = al.regulariser.parameter.DiffusionRegulariser(global_fix_level.spacing)
                diffreg.set_weight(self.config['hyperparameter']['global_r_w'])
                reg.append(diffreg)

        if self.config['method']['is_other_reg']:
            mmask = mask_pyramid['other'][level].image
            other_reg = al.regulariser.parameter.MaskDiffusionRegulariser(global_fix_level.spacing,mask = mmask)
            other_reg.set_weight(self.config['hyperparameter']['other_r_w'])
            reg.append(other_reg)

        if self.config['method']['is_spine_reg']:
            mmask = mask_pyramid['spine'][level].image
            spine_reg = al.regulariser.parameter.MaskSparsityRegulariser(global_fix_level.spacing,mask = mmask)
            spine_reg.set_weight(self.config['hyperparameter']['spine_r_w'])
            reg.append(spine_reg)

        if self.config['method']['is_surface_reg']:
            mmask = mask_pyramid['surface'][level].image
            surface_reg = al.regulariser.parameter.MaskIsotropicTVRegulariser(global_fix_level.spacing,mask=mmask)
            surface_reg.set_weight(self.config['hyperparameter']['surface_r_w'])
            reg.append(surface_reg)
        return reg

    def start(self,case):
        """
        start registration
        """
        startTime = 0
        im_pyramid,mask_pyramid,points,size_pyramid = self.loaddata(case=case)
        points_fix, points_mov = points['fix'], points['mov']
        constant_flow = None
        nlevel = self.config['hyperparameter']['nlevel']
        startTime = time.time()
        for level in range(nlevel):
            loss = self.setLoss(im_pyramid,mask_pyramid,level)
            reg = self.setRegularization(im_pyramid,mask_pyramid,level)
            
            constant_flow,current_displacement = self.register(im_pyramid,level,constant_flow,loss,reg,size_pyramid)
            # generate SimpleITK displacement field and calculate TRE
            tmp_displacement = al.transformation.utils.upsample_displacement(current_displacement.clone().to(device=self.device),
                                                                            size_pyramid[-1], 
                                                                            interpolation="linear")

            tmp_displacement = al.transformation.utils.unit_displacement_to_dispalcement(tmp_displacement)  # unit measures to image domain measures
            tmp_displacement = al.Displacement(tmp_displacement,size_pyramid[-1],[1,1,1],[0,0,0]) 
            
    
            # in order to not invert the displacement field, the fixed points are transformed to match the moving points
            
            if using_landmarks:
                print("TRE on that level: "+str(al.Points.TRE(points_mov, al.Points.transform(points_fix, tmp_displacement))))
            del tmp_displacement
            del loss
            del reg
            if level != 0:
                for key in im_pyramid:
                    im_pyramid[key]['mov'][level-1] = 0
                    im_pyramid[key]['fix'][level-1] = 0
                for key in mask_pyramid:
                    mask_pyramid[key][level-1] = 0
                gc.collect()
                th.cuda.empty_cache()

        unit_displacement = current_displacement

        endTime = time.time()
        return unit_displacement,startTime,endTime


    def SetHyperparameter(self,key,value):
        self.config['hyperparameter'][key] = value


    def register(self,im_pyramid,level,constant_flow,loss,reg,size_pyramid):
        """
        registration one level
        """
        print("---- Level "+str(level)+" ----")
        registration = al.PairwiseRegistration()
        transformation = al.transformation.pairwise.BsplineTransformation(size_pyramid[level],
                                                                        sigma=self.config['hyperparameter']['sigma'],
                                                                        order=3,
                                                                        dtype=self.dtype,
                                                                        device=self.device,
                                                                        diffeomorphic=False)
        
    
        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                        size_pyramid[level],
                                                                        interpolation="linear")
            transformation.set_constant_flow(constant_flow)
        registration.set_transformation(transformation)

        
        registration.set_image_loss(loss)
        registration.set_regulariser_parameter(reg)
        optimizer = th.optim.Adam(transformation.parameters(), 
                                lr=self.config['hyperparameter']['step_size'][level],
                                weight_decay=0.0001, 
                                amsgrad=True)
        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(self.config['hyperparameter']['niter'][level])
        registration.start()

        # store current flow field
        constant_flow = transformation.get_flow()
        current_displacement = transformation.get_displacement()
        return constant_flow, current_displacement
        

    def evaluation(self,displacement,startTime,endTime,case):
        """
        evaluate result
        Case, TRE, Bone_DICE, Lung_DICE, Jacobian, Time
        """
        loader = Dataloader(self.config['dataset']['dataset'], self.dataconfigpath,True,True)
        points_fix = loader.GetPtsData(case=case,T='00',spacing=loader.GetVolSpacing(case))
        points_mov = loader.GetPtsData(case=case,T='50',spacing=loader.GetVolSpacing(case))
        bone_fix = loader.GetBoneMask(case=case,T='00',voltype='sitk')
        bone_mov = loader.GetBoneMask(case=case,T='50',voltype='sitk')
        lung_fix = loader.GetLungMask(case=case,T='00',voltype='sitk')
        lung_mov = loader.GetLungMask(case=case,T='50',voltype='sitk')

        


        
        Cost_Time = self.GetTime(startTime,endTime)
        Init_Bone_Dice,Bone_Dice = self.GetBoneDice(bone_fix,bone_mov,displacement)
        Init_Lung_Dice,Lung_Dice = self.GetLungDice(lung_fix,lung_mov,displacement)
        #注意转换函数会把原来的dis也改变，本质是一片内存区域
        ph_displacement = al.transformation.utils.unit_displacement_to_dispalcement(displacement) # unit measures to image domain measures
        ph_displacement = al.Displacement(displacement, bone_fix.GetSize(),[1,1,1],[0,0,0])

        Init_TRE,TRE = self.GetTRE(points_fix,points_mov,ph_displacement)
        Jacobian = self.GetJacobian(ph_displacement)

        print("TRE: ",Init_TRE,TRE)
        print("Bone Dice: ",Init_Bone_Dice,Bone_Dice)
        print("Lung Dice: ",Init_Lung_Dice,Lung_Dice)
        print("Jacobian: ",Jacobian)
        print("Time: ",Cost_Time)

        evaluation = [case,Init_TRE,TRE,Init_Bone_Dice,Bone_Dice,Init_Lung_Dice,Lung_Dice,Jacobian,Cost_Time]
        """
        evaluation = {}
        evaluation['TRE'] = {'init':Init_TRE,'TRE':TRE}
        evaluation['BoneDice'] = {'init':Init_Bone_Dice,'dice':Bone_Dice}
        evaluation['LungDice'] = {'init':Init_Lung_Dice,'dice':Lung_Dice}
        evaluation['Jacobian'] = Jacobian
        evaluation['Time'] = time
        """

        return evaluation

    def GetTRE(self,points_fix,points_mov,displacement):
        fixed_points_transformed = al.Points.transform(points_fix, displacement)
        TRE = al.Points.TRE(points_mov, fixed_points_transformed,sp=[0.97,0.97,2.5])
        InitTRE = al.Points.TRE(points_mov, points_fix)
        return round(InitTRE,6),round(TRE,6)

    def GetBoneDice(self,bone_fix,bone_mov,displacement):
        bone_fix = sitk.GetArrayFromImage(bone_fix)
        InitDice = mut.dice(bone_fix,sitk.GetArrayFromImage(bone_mov))
        bone_mov = al.Image(bone_mov,th.float64,self.device)
        warp = al.transformation.utils.warp_image(bone_mov,displacement).numpy()
        warp = np.transpose(warp,[2,1,0])
        warp[warp>0] = 1
        warp[warp<1] = 0
        Dice = mut.dice(warp,bone_fix)
        return round(InitDice[0],6),round(Dice[0],6)
    
    def GetLungDice(self,lung_fix,lung_mov,displacement):
        lung_fix = sitk.GetArrayFromImage(lung_fix)
        InitDice = mut.dice(lung_fix,sitk.GetArrayFromImage(lung_mov))
        lung_mov = al.Image(lung_mov,th.float64,self.device)
        warp = al.transformation.utils.warp_image(lung_mov,displacement).numpy()
        warp = np.transpose(warp,[2,1,0])
        warp[warp>0] = 1
        warp[warp<1] = 0
        Dice = mut.dice(warp,lung_fix)
        return round(InitDice[0],6),round(Dice[0],6)

    def GetJacobian(self,displacement):
        scalarImg = sitk.DisplacementFieldJacobianDeterminant(displacement.to(th.float64).itk())
        scalarImg = sitk.GetArrayFromImage(scalarImg)
        scalarImg[scalarImg<=0] = -1
        negative = -scalarImg[scalarImg==-1].sum()
        size = scalarImg.shape[0]*scalarImg.shape[1]*scalarImg.shape[2]
        return round(negative/size,6)

    def GetTime(self,startTime,endTime):
        return round(endTime - startTime,6)

    def logger(self,result,postfix):
        """
        save log
        """
        logRoot = os.path.join(self.appRoot,'log')
        if not os.path.exists(logRoot):
            os.makedirs(logRoot)
        logname = time.strftime("%Y_%m_%d_%H%M%S",time.localtime(time.time()))
        sublogRoot = os.path.join(logRoot,logname+"_"+postfix)
        if not os.path.exists(sublogRoot):
            os.makedirs(sublogRoot)
        df = pd.DataFrame(result)
        header = ['Case','InitTRE','TRE','InitBoneDice','BoneDice','InitLungDice','LungDice','Jacobian','Time']
        df.to_csv(os.path.join(sublogRoot,'result.csv'),header=header,index=None)
        jsonstr = json.dumps(self.config,indent=1)
        with open(os.path.join(sublogRoot,'config.json'),"w") as f:
            f.write(jsonstr)

    def startAll(self):
        result = []
        for i in range(1,2):
            case = str(i)
            unit_displacement,startTime,endTime = self.start(case)
            evalueation = self.evaluation(unit_displacement.to(th.float64),startTime,endTime,case)
            result.append(evalueation)
            del unit_displacement
        self.logger(result,'myglobal')

    def startAllAnalysisSigmas(self):
        randid = str(random.randint(0,99999))
        result = []
        sigmas = [
            [12,12,12],
            [11,11,11],
            [10,10,10],
            [9,9,9],
            [8,8,8],
            [7,7,7],
            [6,6,6],
        ]
        for sigma in sigmas:
            self.config['hyperparameter']['sigma'] = sigma
            s = [str(i) for i in sigma]
            postfix = 'sigmas_'+randid+'_'.join(s)
            for i in range(1,11):
                case = str(i)
                unit_displacement,startTime,endTime = self.start(case)
                evalueation = self.evaluation(unit_displacement.to(th.float64),startTime,endTime,case)
                result.append(evalueation)
                del unit_displacement

            self.logger(result,postfix)
            result = []

    def startAllAnalysisReg(self):
        randid = str(random.randint(0,99999))
        result = []
        Reg = [
            #0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6
            #0.001,0.005,0.6,
            0.0001,0.0005,0.0008
        ]
        for r in Reg:
            self.config['hyperparameter']['surface_r_w'] = r
            self.config['hyperparameter']['spine_r_w'] = r
            self.config['hyperparameter']['other_r_w'] = r
            postfix = 'regs_'+randid+'_'+str(r)
            for i in range(1,11):
                case = str(i)
                unit_displacement,startTime,endTime = self.start(case)
                evalueation = self.evaluation(unit_displacement.to(th.float64),startTime,endTime,case)
                result.append(evalueation)
                del unit_displacement

            self.logger(result,postfix)
            result = []

    def startAllAnalysisLevel(self):
        randid = str(random.randint(0,99999))
        
        result = []
        nlevels=[
            3,4,5,6
        ]
        pyramids=[
            [[4,4,4],[2,2,2]],
            [[8,8,8],[4,4,4],[2,2,2]],
            [[16,16,16],[8,8,8],[4,4,4],[2,2,2]],
            [[32,32,32],[16,16,16],[8,8,8],[4,4,4],[2,2,2]]
        ]
        niters = [
            [800,100,50],
            [800,800,100,50],
            [800,800,800,100,50],
            [800,800,800,800,100,50],
            ]
        for i in range(len(nlevels)):
            self.config['hyperparameter']['nlevel'] = nlevels[i]
            self.config['hyperparameter']['pyramid'] = pyramids[i]
            self.config['hyperparameter']['niter'] = niters[i]

            postfix = 'levels_'+randid+'_'+str(nlevels[i])
            for i in range(1,11):
                case = str(i)
                unit_displacement,startTime,endTime = self.start(case)
                evalueation = self.evaluation(unit_displacement.to(th.float64),startTime,endTime,case)
                result.append(evalueation)
            self.logger(result,postfix)
            result = []

    def startAllAnalysisBoneBody(self):
        randid = str(random.randint(0,99999))
        result = []
        Bone = [
            0.001,0.005,0.01,0.03,0.05,0.07,0.09,0.1,0.2,0.3,0.4,0.5,0.6
        ]
        for r in Bone:
            self.config['hyperparameter']['bone_w'] = r
            self.config['hyperparameter']['body_w'] = r
            postfix = 'bonebody_'+randid+'_'+str(r)
            for i in range(1,11):
                case = str(i)
                unit_displacement,startTime,endTime = self.start(case)
                evalueation = self.evaluation(unit_displacement.to(th.float64),startTime,endTime,case)
                result.append(evalueation)
                del unit_displacement

            self.logger(result,postfix)
            result = []



def main():
    configpath = 'F:\\Code_git\\src\\config_xin.json'
    dataconfigpath = 'F:\\Code_git\\src\\dataset.json'
    appRoot = 'F:\\Code_git\\src'
    registration = Registration(configpath,dataconfigpath,appRoot)

    #registration.startAllAnalysisSigmas()
    #registration.startAllAnalysisReg()
    #registration.startAllAnalysisBoneBody()
    registration.startAll()
    print("=================================================================")
    #print("Registration done in: ", end - start, " seconds")

if __name__ == '__main__':
    #saveMidData()
    main()


