import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
# ================================
# 3GPP Rural Macro (RMa) Building Distribution Model
# ================================

D = 4  # 默认区域边长 D x D km²（通过 set_region 可配置；对齐 config 单位约定：坐标/10 转 km）
step = 101  # 分辨率为 D / (step-1) km

# ============== 乡村建筑参数（与区域大小无关） ==============
ALPHA = 0.12  # 建筑物占地系数（乡村更稀疏：0.2→0.12）
BETA = 100    # 建筑物密度参数（乡村：更多更小，Side≈35m）
GAMA = 6      # 建筑物高度分布（乡村低矮：Rayleigh 中位 ~7m）
MAXHeight = 12
MINHeight = 4

# 基站位置（作为 D 的比例，随区域/基站数量缩放；Bug 3 修复：不再硬编码 4km 坐标）
# N=2/3/4 → 4/5/6 个基站（密度约恒定，保持中断率可比）
_BS_FRAC = {
    4: [[0.2385, 0.228], [0.7445, 0.2445], [0.2315, 0.689], [0.674, 0.7445]],  # 原 SingleUAV 2×2
    5: [[0.2385, 0.228], [0.7445, 0.2445], [0.2315, 0.689], [0.674, 0.7445], [0.5, 0.5]],  # 四角 + 中心
    6: [[0.2, 0.3], [0.5, 0.3], [0.8, 0.3], [0.2, 0.7], [0.5, 0.7], [0.8, 0.7]],  # 2×3 网格
}

BS_loc = None
terrain_height = None
HeightMapMatrix = None
HeighMapArray = None


def set_region(D_km, n_bs=4, map_seed=None):
    """按区域边长 D_km (km) 重新生成地形/建筑/基站。

    ``map_seed`` 显式控制地形/建筑随机性（None = 保持旧默认 42/1）；使用局部
    ``RandomState``，不修改全局 NumPy RNG。更新 ``D`` / ``BS_loc`` / ``HeighMapArray``
    / ``terrain_height``；checkLoS / getPointOutageMatrix 自动使用新区域。
    """
    global D, BS_loc, terrain_height, HeightMapMatrix, HeighMapArray

    D = int(D_km)  # 区域边长须为整数（D*step 作为数组尺寸）

    if map_seed is None:
        terrain_seed, build_seed = 42, 1   # 旧默认，保持可复现
    else:
        terrain_seed, build_seed = int(map_seed), int(map_seed) + 1
    rng_t = np.random.RandomState(terrain_seed)
    rng_b = np.random.RandomState(build_seed)

    # ============== 生成偏远乡村地形（平缓丘陵，起伏 0~50m） ==============
    terrain_noise = rng_t.randn(D * step, D * step)
    terrain_height = gaussian_filter(terrain_noise, sigma=20)  # 平滑成 ~200m 尺度的连绵丘陵
    terrain_height = (terrain_height - terrain_height.min()) / (terrain_height.max() - terrain_height.min())
    terrain_height = terrain_height * 50.0  # 丘陵起伏 0~50m

    # ============== 生成乡村建筑（稀疏低矮，4~12m） ==============
    N = int(BETA * (D ** 2))           # 建筑总数
    A = ALPHA * (D ** 2) / N           # 平均每栋建筑占地面积 (km²)
    Side = np.sqrt(A)                  # 每栋建筑占地边长 (km)

    H_vec = np.clip(rng_b.rayleigh(GAMA, N), MINHeight, MAXHeight)  # 建筑高度采样
    XLOC = rng_b.uniform(0, D, N)  # 建筑中心位置（均匀随机）
    YLOC = rng_b.uniform(0, D, N)

    # ============== 构建最终高度图（地形 + 建筑，重叠取最高不累加） ==============
    BuildingLayer = np.zeros((D * step, D * step))
    for i in range(N):
        x1 = int(np.clip(np.floor((XLOC[i] - Side / 2) * (step - 1)), 0, D * (step - 1)))
        x2 = int(np.clip(np.ceil((XLOC[i] + Side / 2) * (step - 1)), 0, D * (step - 1)))
        y1 = int(np.clip(np.floor((YLOC[i] - Side / 2) * (step - 1)), 0, D * (step - 1)))
        y2 = int(np.clip(np.ceil((YLOC[i] + Side / 2) * (step - 1)), 0, D * (step - 1)))
        BuildingLayer[x1:x2 + 1, y1:y2 + 1] = np.maximum(BuildingLayer[x1:x2 + 1, y1:y2 + 1], H_vec[i])
    HeightMapMatrix = terrain_height + BuildingLayer
    HeighMapArray = HeightMapMatrix.reshape(1, (D * step) ** 2)

    # ============== 基站（随 D 与基站数缩放） ==============
    _frac = np.asarray(_BS_FRAC[n_bs], dtype=float)
    BS_loc = np.column_stack([_frac * D, np.full(n_bs, 0.025)])


set_region(4)  # 默认 4km×4km 区域（N=2）
    

# BS_loc 已改为在 set_region() 内按 D 缩放生成（Bug 3 修复）
# BS_loc=np.array([[1, 1, 0.025], [1.5774,1.333, 0.025], [1, 1.6667,0.025,], [0.4226,1.3333,  0.025],
#                [0.4226, 0.6667,  0.025], [1, 0.3333, 0.025], [1.5774,0.6667,  0.025]]) # 2km
BS_thetaD = 120  # The downtile angle in degree [0, 180]
PB = 0.1 # BS传输功率
Fc = 2  # Operating Frequency in GHz
SIR_THRESHOLD=3 #SIR threshold in dB for outage


LightSpeed = 3 * (10 ** 8)
WaveLength = LightSpeed / (Fc * (10 ** 9))  # wavelength in meter 波长
SectorBoreSiteAngle = [-120, 0, 120]  # the sector angle for each BS每个 BS 的扇区角
Sec_num = np.size(SectorBoreSiteAngle)  # 每个小区的扇区数
FastFadingSampleSize = 1000  # number of signal measurements per time step每个时间步的信号测量次数


# 存储天线增益
def getAntennaGain(Theta_deg, Phi_deg):
    # Basic Setting about Antenna Arrays
    ArrayElement_Horizontal = 1  # number of array elements in horizontal水平方向的数组元素数
    ArrayElement_Vertical = 8
    DV = 0.5 * WaveLength  # spacing for vertical array垂直阵列的间距
    DH = 0.5 * WaveLength  # spacing for horizontal array水平阵列间距
    angleTiltV = BS_thetaD
    angleTiltH = 0  # horizontal tilt angle水平倾斜角
    # Find the element power gain求元素功率增益
    angle3dB = 65
    Am = 30
    AH = -np.min([12 * (Phi_deg / angle3dB) ** 2, Am])  # element power gain in horizontal水平元件功率增益
    AV = -np.min([12 * ((Theta_deg - 90) / angle3dB) ** 2, Am])  # 垂直元件功率增益
    Gemax = 8  # dBi antenna gain in dB above an isotropic radiator, Maximum directional gain of an antenna elementdBi 天线增益，以 dB 为单位，高于各向同性辐射器，天线元件的最大方向增益
    Aelement = -np.min([-(AH + AV), Am])
    GelementdB = Gemax + Aelement  # dBi
    Gelement = 10 ** (GelementdB / 10)
    Felement = np.sqrt(Gelement)
    # Find array gain
    k = 2 * np.pi / WaveLength  # wave number
    kVector = k * np.array([np.sin(Theta_deg / 180 * np.pi) * np.cos(Phi_deg / 180 * np.pi),
                            np.sin(Theta_deg / 180 * np.pi) * np.sin(Phi_deg / 180 * np.pi),
                            np.cos(Theta_deg / 180 * np.pi)])  # wave vector
    rMatrix = np.zeros(shape=(ArrayElement_Horizontal * ArrayElement_Vertical, 3))
    for n in range(ArrayElement_Horizontal):
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 2] = np.arange(ArrayElement_Vertical) * DV
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 1] = n * DH
    SteeringVector = np.exp(-1j * (rMatrix.dot(np.transpose(kVector))))
    # Vertical Weight Vector
    Weight_Vertical = (1 / np.sqrt(ArrayElement_Vertical)) * np.exp(
        -1j * k * np.arange(ArrayElement_Vertical) * DV * np.cos(angleTiltV / 180 * np.pi))
    Weight_Horizontal = (1 / np.sqrt(ArrayElement_Horizontal)) * np.exp(
        -1j * k * np.arange(ArrayElement_Horizontal) * DH * np.sin(angleTiltH / 180 * np.pi))
    Weight2D = np.kron(Weight_Horizontal, np.transpose(Weight_Vertical))
    WeightFlatten = Weight2D.reshape(1, ArrayElement_Vertical * ArrayElement_Horizontal)
    ArrayFactor = np.conjugate(WeightFlatten).dot(SteeringVector.reshape(ArrayElement_Vertical, 1))
    Farray = Felement * ArrayFactor
    Garray = (np.abs(Farray)) ** 2
    return 10 * np.log10(Garray).item(), Farray

#=========Main Function that determines the best outage from all BS at a given location=======
#loc_vec: a matrix, nx3, each row is a (x,y,z) location
#SIR_th: the SIR threshold for determining outage
def getPointMiniOutage(loc_vec):
    numLoc=len(loc_vec)
    Out_vec=[]
    Out_vec_SINR=[]
    for i in range(numLoc):
        PointLoc=loc_vec[i,:]
        OutageMatrix,OutageMatrixSINR=getPointOutageMatrix(PointLoc,SIR_THRESHOLD)
        MiniOutage=np.min(OutageMatrix)
        MiniOutageSINR = np.max(OutageMatrixSINR)
        Out_vec.append(MiniOutage)
        Out_vec_SINR.append(MiniOutageSINR)
    return Out_vec,Out_vec_SINR

#For a given location, return the empirical outage probaibility from all sectors of all BSs
#PointLoc:  the given point location
#SIR_th: the SIR threshold for defining the outage 
#OutageMatrix: The average outage probability for connecting with each site, obtained by averaging over all the samples

def getPointOutageMatrix(PointLoc,SIR_th):
    numBS=len(BS_loc)
    SignalFromBS=[]
    TotalPower=0

    for i in range(len(BS_loc)):
        BS=BS_loc[i,:]
        LoS=checkLoS(PointLoc,BS)
        MeasuredSignal=getReceivedPower_RicianAndRayleighFastFading(PointLoc,BS,LoS)
        SignalFromBS.append(MeasuredSignal)
        TotalPower=TotalPower+MeasuredSignal
    TotalPowerAllSector=np.sum(TotalPower, axis=1) #the interference of all power
    OutageMatrix=np.zeros(shape=(numBS,Sec_num))
    SIR_dB_avg = np.zeros(shape=(numBS,Sec_num))
    
    for i in range(len(BS_loc)):
        SignalFromThisBS=SignalFromBS[i]
        for sector in range(Sec_num):
            SignalFromThisSector=SignalFromThisBS[:,sector]
            #--------------------主要改这一项-----------------------#
            SIR=SignalFromThisSector/(TotalPowerAllSector-SignalFromThisSector+10 ** ((-140 - 30) / 10)) # SIR; 对齐 SingleUAV 单机代码（干扰主导，噪声底可忽略）
            SIR_dB=10*np.log10(SIR)
            SIR_dB_avg[i, sector] = np.sum(SIR_dB) / len(SIR_dB)
            # print(np.sum(SIR_dB<SIR_th))
            # print(len(SIR_dB))
            OutageMatrix[i,sector]=np.sum(SIR_dB<SIR_th)/len(SIR_dB)
            # print(OutageMatrix[i,sector])
    return OutageMatrix,SIR_dB_avg



def getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg, LoS, Fc=2.0, H_u=100, SectorBoreSiteAngle=[0, 120, -120]):
    """
    Calculate large-scale received power (linear scale) from BS using RMa model (3GPP TR 36.814)

    Parameters:
        PointLoc: [x_uav, y_uav, h_uav] in km
        BS: [x_bs, y_bs, h_bs] in km
        Theta_deg: elevation angle (for antenna)
        Phi_deg: azimuth angle (for antenna)
        LoS: boolean indicating if LoS exists
        Fc: carrier frequency in GHz
        H_u: UAV height in meters
        SectorBoreSiteAngle: list of sector directions

    Returns:
        Prx: received power (linear scale)
    """
    Sector_num = len(SectorBoreSiteAngle)
    Phi_Sector_ref = Phi_deg - np.array(SectorBoreSiteAngle)
    Phi_Sector_ref[Phi_Sector_ref < -180] += 360
    Phi_Sector_ref[Phi_Sector_ref > 180] -= 360

    ChGain_dB = np.zeros((1, Sector_num))
    for i in range(Sector_num):
        ChGain_dB[0, i], _ = getAntennaGain(Theta_deg, Phi_Sector_ref[i])
    ChGain=np.power(10,ChGain_dB/10)  # convert from dB to linear

    # Compute 3D distance in meters
    dx = (BS[0] - PointLoc[0]) 
    dy = (BS[1] - PointLoc[1]) 
    dz = (BS[2] - PointLoc[2]) 
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)*1000  # in meters

    H_m = BS[2] * 1000  # BS height in meters

    # PL^{LoS} according to your formula
    PL_LoS = 20 * np.log10(40 * np.pi * d * Fc / 3) + \
             min(0.03 * H_u ** 1.72, 10) * np.log10(d) - \
             0.58 * np.log10(H_m)
             
    PL_NLoS = 161.04 - 7.1 * np.log10(20) + 7.5 * np.log10(H_m) - \
                       (24.37 - 3.7 * (H_m / H_u) ** 2) * np.log10(H_m) + \
                       (43.42 - 3.1 * np.log10(H_m)) * (np.log10(d) - 3) + \
                       20 * np.log10(Fc) - (3.2 * (np.log10(11.75 * H_m)) ** 2 - 4.97)
    if LoS:
        PathLoss_dB = PL_LoS
    else:      
        PathLoss_dB = PL_NLoS

    # Convert path loss to linear scale
    PathLoss_linear = 10 ** (-PathLoss_dB / 10)

    # Compute received power (you need to define PB externally)
    Prx = ChGain * PB * PathLoss_linear  # [1, Sector_num]

    return Prx
 
# Return the received power at a location from all the three sectors of a BS
# While the large scale path loss power is a constant for given location and site, the fast fading may change very fast.
# Hence, we return multiple fast fading coefficients. The number of samples is determined by FastFadingSampleSize
# A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
def getReceivedPower_RicianAndRayleighFastFading(PointLoc, BS, LoS):
    HorizonDistance = np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2) + 1e-9  # 防 UAV 正上方除零
    Theta = np.arctan((BS[2] - PointLoc[2]) / HorizonDistance)  # elevation angle
    Theta_deg = np.rad2deg(Theta) + 90  # convert to the (0,180) degree
    if (PointLoc[1] == BS[1]) & (PointLoc[0] == BS[0]):
        Phi = 0
    else:
        Phi = np.arctan((PointLoc[1] - BS[1]) / (PointLoc[0] - BS[0] + 0.00001))  # to avoid dividing by 0
    Phi_deg = np.rad2deg(Phi)
    # Convert the horizontal degree to the range (-180,180)
    if (PointLoc[1] > BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg + 180
    elif (PointLoc[1] < BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg - 180
    LargeScale = getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg,
                                          LoS)  # large-scale received power based on path loss

    # the random component, which is Rayleigh fading.
    # Deterministic: seed a LOCAL RNG from the (rounded) point + BS coordinates so
    # the fast fading is a FIXED function of location, independent of the caller's
    # global RNG state (reproducible pre-constructed radio map).
    _seed = int(abs(hash((round(float(PointLoc[0]), 3), round(float(PointLoc[1]), 3),
                          round(float(BS[0]), 3), round(float(BS[1]), 3)))) % (2 ** 31))
    _rng = np.random.RandomState(_seed)
    RayleighComponent = np.sqrt(0.5) * (
                _rng.randn(FastFadingSampleSize, 3) + 1j * _rng.randn(FastFadingSampleSize, 3))  # (1000,3)
    # print("瑞丽数组大小",RayleighComponent.shape)

    if LoS:  # LoS, fast fading is given by Rician fading with K factor 15 dB
        K_R_dB = 15  # Rician K factor in dB
        K_R = 10 ** (K_R_dB / 10)
        threeD_distance = 1000 * np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2 + (
                    BS[2] - PointLoc[2]) ** 2)  # 3D distance in meter
        DetermComponent = np.exp(-1j * 2 * np.pi * threeD_distance / WaveLength)  # deterministic component
        AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) * DetermComponent + np.sqrt(1 / (K_R + 1)) * RayleighComponent
    else:  # NLoS, fast fading is Rayleigh fading
        AllFastFadingCoef = RayleighComponent

    h_overall = AllFastFadingCoef * np.sqrt(np.tile(LargeScale, (FastFadingSampleSize, 1)))
    PowerInstant = np.abs(h_overall) ** 2  # the instantneous received power in Watt
    # print("瞬时功率",PowerInstant.shape)
    return PowerInstant

# This function check whether there is LoS between the BS and the given Loc三维坐标
# 概率LoS模型
# def checkLoS(PointLoc, BS):

#     # Calculate 3D distance
#     dx = PointLoc[0] - BS[0]
#     dy = PointLoc[1] - BS[1]
#     dz = PointLoc[2] - BS[2]
#     d = np.sqrt(dx**2 + dy**2 + dz**2)*1000

#     # LoS probability
#     if d < 10:
#         p_los = 1.0
#     else:
#         p_los = np.exp(-d/ 1000.0)

    
#     p = [p_los, 1 - p_los]
#     # Randomly choose LoS or NLoS based on the probability
#     LoS = np.random.choice([True, False], p=p)  
#     print(f"LoS probability: {p_los}, LoS: {LoS}")
#     return LoS

# 原论文中的 LoS 检查函数
def checkLoS(PointLoc, BS):
    SamplePoints = np.linspace(0, 1, 100) # (0,1)100个点
    XSample = BS[0] + SamplePoints * (PointLoc[0] - BS[0])
    YSample = BS[1] + SamplePoints * (PointLoc[1] - BS[1])
    ZSample = BS[2] + SamplePoints * (PointLoc[2] - BS[2])
    XRange = np.floor(XSample * (step - 1)) # 地板除
    YRange = np.floor(YSample * (step - 1))
    XRange = [max(x, 0) for x in XRange]  # remove the negative idex去掉负号
    YRange = [max(x, 0) for x in YRange]  # remove the negative idex
    Idx_vec = np.int_((np.array(XRange) * D * step + np.array(YRange))) # 高度数组里的下标
    SelectedHeight = [HeighMapArray[0, i] for i in Idx_vec]  # 比较高度
    if any([x > y * 1000 for (x, y) in zip(SelectedHeight, ZSample)]): # 建筑物高度(m)大于样点高度(km→m)
        # print("NLoS")
        return False
    else:
        # print("LoS")
        return True

def getPointDateRate(loc_vec):  # 单点查询：返回该点的最大 SINR(dB) 标量（多行会丢数据，故拒绝多行）
    numLoc = len(loc_vec)  # numLoc = n
    if numLoc != 1:
        raise ValueError(
            "getPointDateRate 只支持单点输入 (shape (1,3))，收到 %d 个点；请逐点查询" % numLoc)
    Out_vec = []
    Out_SINR_vec = []
    
    for i in range(numLoc):  # 循环n次
        PointLoc = loc_vec[i, :]
        OutageMatrix,SIR_dB_avg = getPointOutageMatrix(PointLoc, SIR_THRESHOLD)
        MiniOutage = np.min(OutageMatrix) #取出中断概率最小的基站的信号
        MaxSINR = np.max(SIR_dB_avg)
        Out_SINR_vec.append(MaxSINR)  # 1x201数组
        # 吞吐量
    return MaxSINR #返回的是信噪比最大的值   

##============VIew the radio map for given height
if __name__ == '__main__':
   
    step = 101
    UAV_height=0.100 # UAV height in km
    X_vec=range(D*(step-1)+1)
    Y_vec=range(D*(step-1)+1)
    numX,numY=np.size(X_vec),np.size(Y_vec)

    OutageMapActual=np.zeros(shape=(numX,numY))
    OutageMapActualSINR=np.zeros(shape=(numX,numY))
    Loc_vec_All=np.zeros(shape=(numX*numY,3))

    for i in range(numX):
        Loc_vec=np.zeros(shape=(numY,3))
        Loc_vec[:,0]=X_vec[i]/step
        Loc_vec[:,1]=np.array(Y_vec)/step
        Loc_vec[:,2]=UAV_height
        Loc_vec_All[i*numY:(i+1)*numY,:]=Loc_vec
        
        OutageMapActual[:,i],OutageMapActualSINR[:,i] = getPointMiniOutage(Loc_vec)
        
    Outage_vec_All = np.reshape(OutageMapActual, numX * numY)

   
    fig = plt.figure(10)
    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, OutageMapActualSINR)
    v = np.arange(-10, 24, 2)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('SINR', labelpad=20, rotation=270, fontsize=14)
    plt.xlabel('x (meter)', fontsize=14)
    plt.ylabel('y (meter)', fontsize=14)
    plt.savefig('results/figs/OutageMapActualSINR.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    fig = plt.figure(11)
    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1-OutageMapActual)
    v = np.linspace(0, 1, 11, endpoint=True)    
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    plt.xlabel('x (meter)', fontsize=14)
    plt.ylabel('y (meter)', fontsize=14)
    plt.savefig('results/figs/OutageMapActual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    np.savez('results/datas/Radio_datas', OutageMapActual, OutageMapActualSINR ,X_vec, Y_vec)