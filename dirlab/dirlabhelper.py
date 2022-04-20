import os
import SimpleITK as sitk
import numpy as np
import torch as th
import json

class Dataloader():
    """
    load data
    """
    def __init__(self,dataset,dataConfigRoot,crop,resample):
        self.dataset = dataset
        self.crop = crop
        self.resample = resample
        self.dataConfig = None
        with open(dataConfigRoot, 'r') as f:
            self.dataConfig = json.load(f)[dataset]


    def __GetVolPath(self,case='1',T='00'):
        root = self.dataConfig['root']
        if self.dataset=='dirlab':
            if self.resample:
                volpath = os.path.join(root,"Case{0}Pack".format(case),"Resample","C{0}T{1}_r.mha".format(case,T))
            else:
                volpath = os.path.join(root,"Case{0}Pack".format(case),"Mha","C{0}T{1}.mha".format(case,T))
        elif self.dataset=='copd':
            if T=='00':
                volpath = os.path.join(root,"copd{0}".format(case),'Resample','copd{0}_iBHCT.mha'.format(case))
            elif T=='50':
                volpath = os.path.join(root,"copd{0}".format(case),'Resample','copd{0}_eBHCT.mha'.format(case))
        return volpath


    def __GetPtsPath(self,case='1',T='00'):
        root = self.dataConfig['root']
        if self.dataset=='dirlab':
            return os.path.join(root,"Case{0}Pack".format(case),"Pts","C{0}T{1}_300.pts".format(case,T))
        elif self.dataset=='copd':
            if T=='00':
                return os.path.join(root,"copd{0}".format(case),'Pts',"copd{0}_300_iBH_xyz_r1.txt".format(case,T))
            elif T=='50':
                return os.path.join(root,"copd{0}".format(case),'Pts',"copd{0}_300_eBH_xyz_r1.txt".format(case,T))


    def __GetLungMaskPath(self,case='1',T='00'):
        root = self.dataConfig['root']
        if self.dataset=='dirlab':
            return os.path.join(root,"Case{0}Pack".format(case),"LungMask","C{0}T{1}_r_lung_mask.mha".format(case,T))
        elif self.dataset=='copd':
            if T=='00':
                return os.path.join(root,"copd{0}".format(case),'LungMask',"copd{0}_iBHCT_mask.mha".format(case))
            elif T=='50':
                return os.path.join(root,"copd{0}".format(case),'LungMask',"copd{0}_eBHCT_mask.mha".format(case))

    

    def __GetBoneMaskPath(self,case='1',T='00'):
        root = self.dataConfig['root']
        if self.dataset=='dirlab':
            return os.path.join(root,"Case{0}Pack".format(case),"BoneMask","C{0}T{1}_r_bone_mask.mha".format(case,T))
        elif self.dataset=='copd':
            if T=='00':
                return os.path.join(root,"copd{0}".format(case),'BoneMask',"copd{0}_iBHCT_bonemask.mha".format(case,T))
            elif T=='50':
                return os.path.join(root,"copd{0}".format(case),'BoneMask',"copd{0}_eBHCT_bonemask.mha".format(case,T))


    def GetVolSpacing(self, case='1'):
        return self.dataConfig['spacing'][int(case)-1]


    def GetVolShape(self, case='1'):
        return self.dataConfig['size'][int(case)-1]


    def GetVolumeData(self,case='1',T='00',voltype='numpy',dtype='int16',offset=0,ctmin=50,ctmax=2500):
        """
        对数据进行裁剪，数据类型转换，minmax转换
        """
        path = self.__GetVolPath(case=case,T=T)
        vol = sitk.ReadImage(path)
        rawdata = sitk.GetArrayFromImage(vol).astype(dtype)+offset
        rawdata[rawdata>=ctmax]=ctmax
        rawdata[rawdata<=ctmin] = ctmin
        volcrop = self.dataConfig['resampleCrop']
        if self.crop:
            c = int(case)
            x1,x2 = volcrop[c-1][0][0],volcrop[c-1][0][1]
            y1,y2 = volcrop[c-1][1][0],volcrop[c-1][1][1]
            z1,z2 = volcrop[c-1][2][0],volcrop[c-1][2][1]
            rawdata = rawdata[z1:z2,y1:y2,x1:x2].copy()
        if voltype == 'numpy':
            return rawdata
        elif voltype=='torch':
            return th.tensor(rawdata)
        elif voltype == 'sitk':
            newvol = sitk.GetImageFromArray(rawdata)
            newvol.SetSpacing(vol.GetSpacing())
            newvol.SetOrigin(vol.GetOrigin())
            newvol.SetDirection(vol.GetDirection())
            return newvol
        else:
            raise Exception("Invalid dtype!")


    def GetPtsData(self, case='1',T='00',spacing=[1,1,1]):
        path = self.__GetPtsPath(case=case,T=T)
        m = [0,0,0]
        if self.crop:
            volcrop = self.dataConfig['resampleCrop']
            c = int(case)
            m = [volcrop[c-1][0][0],volcrop[c-1][1][0],volcrop[c-1][2][0]]
        return  np.loadtxt(path).astype('float32')*spacing - m

    def __GetPtsMinMax(self,pts):
        pmax = pts.max(axis=0)
        pmin = pts.min(axis=0)
        return pmax,pmin


    def GetLungMask(self, case='1',T='00',voltype='numpy',dtype='int16'):
        path = self.__GetLungMaskPath(case=case,T=T)
        vol = sitk.ReadImage(path)
        rawdata = sitk.GetArrayFromImage(vol).astype(dtype)
        rawdata[rawdata>=1]=1
        if self.crop:
            volcrop = self.dataConfig['resampleCrop']
            c = int(case)
            x1,x2 = volcrop[c-1][0][0],volcrop[c-1][0][1]
            y1,y2 = volcrop[c-1][1][0],volcrop[c-1][1][1]
            z1,z2 = volcrop[c-1][2][0],volcrop[c-1][2][1]
            rawdata = rawdata[z1:z2,y1:y2,x1:x2].copy()
        if voltype == 'numpy':
            return rawdata
        elif voltype=='torch':
            return th.tensor(rawdata)
        elif voltype == 'sitk':
            newvol = sitk.GetImageFromArray(rawdata)
            newvol.SetSpacing(vol.GetSpacing())
            newvol.SetOrigin(vol.GetOrigin())
            newvol.SetDirection(vol.GetDirection())
            return newvol
        else:
            raise Exception("Invalid dtype!")
    

    def GetBoneMask(self,case='1',T='00',voltype='numpy',dtype='int16'):
        path = self.__GetBoneMaskPath(case=case,T=T)
        vol = sitk.ReadImage(path)
        rawdata = sitk.GetArrayFromImage(vol).astype(dtype)
        
        if self.crop:
            c = int(case)
            volcrop = self.dataConfig['resampleCrop']
            x1,x2 = volcrop[c-1][0][0],volcrop[c-1][0][1]
            y1,y2 = volcrop[c-1][1][0],volcrop[c-1][1][1]
            z1,z2 = volcrop[c-1][2][0],volcrop[c-1][2][1]
            rawdata = rawdata[z1:z2,y1:y2,x1:x2].copy()
        if voltype == 'numpy':
            return rawdata
        elif voltype=='torch':
            return th.tensor(rawdata)
        elif voltype == 'sitk':
            newvol = sitk.GetImageFromArray(rawdata)
            newvol.SetSpacing(vol.GetSpacing())
            newvol.SetOrigin(vol.GetOrigin())
            newvol.SetDirection(vol.GetDirection())
            return newvol
        else:
            raise Exception("Invalid dtype!")