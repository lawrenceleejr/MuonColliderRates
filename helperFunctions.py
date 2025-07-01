import numpy as np
import seaborn as sns
import ROOT


def doFillBetween(x,y,axis,n=10,dy=1,color="k",alpha=0.03,log=True,clip_on=True):
    initialY = y
    tmpy = initialY
    print(x,y)

    colorpal = sns.light_palette(color, n)[::-1]
    for i in range(n):
        if log:
            axis.fill_between(x,tmpy, [thing*dy for thing in tmpy],linewidth=0,color=colorpal[i],alpha = alpha*((n-i)/float(n) ) ,  clip_on=clip_on)
            tmpy = [thing*dy for thing in tmpy]
        # else:
        # 	axis.fill_between(x,tmpy, [thing*dy for thing in tmpy],linewidth=0,color=colorpal[i],alpha = alpha*((n-i)/float(n) ) )
        # 	tmpy = [thing*dy for thing in tmpy]


def getArraysFromTGraph(tgraph):
    xArray, yArray = [],[]
    for iPoint in range(tgraph.GetN()):
        x,y = ROOT.Double(0), ROOT.Double(0)
        # print (x,y)
        tgraph.GetPoint(iPoint,x,y)
        xArray.append(x)
        yArray.append(y)
    # print (xArray)
    return xArray,yArray




def add_box_endpoints(arr, point=1e-8):
    # Create the (0, 0) point with same dtype
    points = np.array([
		(arr[-1][0],arr[-1][1]),
		(arr[-1][0],point),
		(arr[0][0],point),
		], dtype=arr.dtype)
    
    # Concatenate zero, original array, zero
    return np.concatenate([points, arr])


def add_box_endpoints_y(arr, point=1e-8):
    # Create the (0, 0) point with same dtype
    points = np.array([
		(arr[-1][0],arr[-1][1]),
		(point,arr[-1][1]),
		(point,arr[0][1]),
		], dtype=arr.dtype)
    
    # Concatenate zero, original array, zero
    return np.concatenate([points, arr])

# https://arxiv.org/pdf/1810.12602
def lifetimeToDm(lifetime):
	return 0.93*0.1/np.power(lifetime,1/3)

def dmToLifetime(dm):
	return np.power(0.93/dm,3)*1e-3

def arrLifetimeToDm(arr):
      return [lifetimeToDm(x) for x in arr]



def add_zero_endpoints(arr, point=(0,0)):
    # Create the (0, 0) point with same dtype
    zero_point = np.array([point], dtype=arr.dtype)
    
    # Concatenate zero, original array, zero
    return np.concatenate([zero_point, arr, zero_point])

def breathe_logy(ax):
    limy = ax.get_ylim()
    m0 = limy[0] * (1-0.15)
    ax.spines.bottom.set_position(('data', m0))

    limx = ax.get_xlim()
    span = limx[1] - limx[0]
    m0 = limx[0] - span*0.04
    ax.spines.left.set_position(('data', m0))

def breathe_logx(ax):
    limy = ax.get_ylim()
    span = limy[1] - limy[0]
    m0 = limy[0] - span*0.04
    ax.spines.bottom.set_position(('data', m0))

    limx = ax.get_xlim()
    # span = limx[1] - limx[0]
    m0 = limx[0]  * (1-0.15)
    ax.spines.left.set_position(('data', m0))



def breathe_logxy(ax):
    limx = ax.get_xlim()
    # span = limx[1] - limx[0]
    m0 = limx[0]  * (1-0.15)
    ax.spines.left.set_position(('data', m0))
    limy = ax.get_ylim()
    m0 = limy[0] * (1-0.15)
    ax.spines.bottom.set_position(('data', m0))

def breathe_logrighty(ax):

    limx = ax.get_xlim()
    # span = limx[1] - limx[0]
    m0 = limx[1]  * (1+0.15)
    ax.spines.right.set_position(('data', m0))
