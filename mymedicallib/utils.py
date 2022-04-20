import SimpleITK as sitk
import numpy as np
import scipy.io as io
from skimage import measure
import torch as th
def resampleVolume(outspacing,vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0,0,0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0,0,0]
    inputdir = [0,0,0]

    #读取文件的size和spacing信息
    
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    #计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0]*inputspacing[0]/outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1]*inputspacing[1]/outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2]*inputspacing[2]/outspacing[2] + 0.5)

    #设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    newvol = resampler.Execute(vol)
    return newvol

def raw2mha(inpath,size,spacing,intype='uint16',outtype='uint16'):
    """
    parameter:
    inpath:raw file path
    outpath:raw out file path
    size:raw file size(z,y,x) such as (94,256,256)
    spacing:raw file pixel spacing.
    intype:raw file data type,default is uint16
    """
    #利用np从文件读取文件
    data = np.fromfile(inpath,dtype=intype)
    
    #reshape数据，这里要注意读入numpy的时候，对应是(z,y,x)
    data = data.reshape(size)
    #data[data==-2000]=0
    #设置输出时的数据类型
    data = data.astype(outtype)
    #转成itk的image
    img:itk.Image = itk.GetImageFromArray(data)
    #设置pixel spacing
    img.SetSpacing(spacing)
    return img


def readmat(path,data='data'):
    rawdata = io.loadmat(path)[data]
    rawdata = rawdata.astype(np.float)
    rawdata = rawdata.transpose([2,1,0,3])
    vol = sitk.GetImageFromArray(rawdata)
    vol.SetSpacing([1.13,1.13,2.5])
    return vol

def getlungsurface(vol,bone):
    dilate = sitk.BinaryDilate(vol,[5,5,5],sitk.sitkBall)
    #sitk.WriteImage(dilate,"E:\\1workspace\\python\\mairlab\\src\\dilatecase1fix.mha")
    erode = sitk.BinaryErode(vol,[1,1,1],sitk.sitkBall)
    #sitk.WriteImage(erode,"E:\\1workspace\\python\\mairlab\\src\\erodecase1fix.mha")
    diff = sitk.GetArrayFromImage(dilate) - sitk.GetArrayFromImage(erode)

    diff = diff.astype(np.int16)
    bone = sitk.GetArrayFromImage(bone).astype(np.int16)

    backbone = bone-diff
    backbone[backbone==-1]=0
    backbone = sitk.GetImageFromArray(backbone)
    diff = sitk.GetImageFromArray(diff)
    return diff,backbone

def getbody(lung):
    lung = sitk.GetArrayFromImage(lung).astype('bool')
    body = ~lung
    body = body.astype(np.int16)
    body = sitk.GetImageFromArray(body)
    return body

def getother(surface,spine):
    surface = th.tensor(sitk.GetArrayFromImage(surface))
    spine = th.tensor(sitk.GetArrayFromImage(spine))
    other = th.ones_like(surface)-surface-spine
    other = sitk.GetImageFromArray(other.cpu().numpy())
    return other


def getbonemask(vol,threshold=1200):
    vol[vol<threshold]=0
    vol[vol>=threshold]=1
    vol = sitk.GetImageFromArray(vol)
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(4)
    bm.SetForegroundValue(1)
    vol = bm.Execute(vol)
    vol = sitk.GetArrayFromImage(vol)
    label = measure.label(vol,connectivity=2)
    props = measure.regionprops(label)

    #计算每个连通域的体素的个数
    for ia in range(len(props)):
        if props[ia].area<=10:
            label[label==props[ia].label]=0
    label[label>1]=1
    return label

def txt2SlicerMarkup(txt,spacing):
    #vtkMRMLMarkupsFiducialNode_0,-272.230,-192.214,160.000,0.000,0.000,0.000,1.000,1,1,0,F-1,,vtkMRMLScalarVolumeNode265
    head = '# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    node = 'vtkMRMLMarkupsFiducialNode_{0},{1},{2},{3},0.000,0.000,0.000,1.000,1,1,0,F-{4},,vtkMRMLScalarVolumeNode265'
    nodelist = []
    for i,p in enumerate(txt):
        x,y,z = -p[0]*spacing[0],-p[1]*spacing[1],p[2]*spacing[2]
        nodelist.append(node.format(i,x,y,z,i+1))
    nodes = '\n'.join(nodelist)
    fcsv = head+nodes
    return fcsv


def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def Dicom2Mha():
    sitk.ReadImage()
    pass
    




