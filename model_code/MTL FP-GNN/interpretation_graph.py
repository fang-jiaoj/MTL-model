import numpy as np
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.pyplot as plt
from fpgnn_ext.model import get_atts_out
from fpgnn_ext.tool import set_intergraph_argument, get_scaler, load_args, load_data, load_model
from fpgnn_ext.data import MoleDataSet
from fpgnn_ext.train import predict
import pandas as pd
import os

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms() #首先，获取分子中的原子总数
    for idx in range( atoms ):
        #使用 GetAtomWithIdx 方法获取分子中的指定索引的原子对象
        # 然后使用 SetProp 方法为原子设置属性 'molAtomMapNumber'，该属性的值设置为原子的索引，通过 str 函数将索引转换为字符串
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol #返回更新后的带有原子索引属性的分子对象

def drawmol_bond(smile,smi_att,path):
    """用于绘制带有键值颜色映射的分子图,接受 SMILES 表示的分子结构、键值权重 smi_att、以及输出图像文件的路径作为输入
    GAT模型的权重系数在键上"""
    smi_att = np.array(smi_att) #将权重矩阵 smi_att 转换为 NumPy 数组
    atom_num = len(smi_att[0]) #获取分子中原子的数量

    #smi_att 矩阵被修改为对称矩阵，以便后续的颜色映射操作可以更准确地表示分子中的键连接
    #将smi_att矩阵的对角线以下的值复制到对角线以上来实现的，故对角线以下是权重，对角线以上是0
    for i in range(atom_num):
        for j in range(i+1):
            smi_att[j][i] = abs(smi_att[j][i]) + abs(smi_att[i][j])
            smi_att[i][j] = 0

    min_value = smi_att.min(axis=(0,1)) #它会在整个矩阵中查找最小值，而不仅仅在每行或每列中查找
    max_value = smi_att.max(axis=(0,1)) #它会在整个矩阵找最大值
    ###用于将权重值映射到颜色映射范围中
    #vmin 和 vmax 分别表示颜色映射的最小值和最大值。
    #max_value+0.15 是为了留出一些余地，以确保最大值的颜色不会变得过饱和
    #创建了一个 ScalarMappable 对象，该对象将用于将矩阵中的值映射到颜色。norm 指定了颜色映射的范围，cmap 参数指定了颜色映射方案
    mol = Chem.MolFromSmiles(smile)
    mol = mol_with_atom_index(mol) ###获得带有原子索引的mol式
    
    bond_list = [] #用于存储要突出显示的键的索引
    atom_list = []
    bond_colors = {} #用于存储键和它们的颜色映射关系
    bond_no = np.nonzero(smi_att)
    bond_atten = []
    #找到 smi_att 矩阵中非零值的索引，即找到有相互作用的原子对的索引，是一个包含两个数组的元组，分别表示原子对的索引

    ##对分子中的化学键进行着色，分子结构中的每个化学键选择合适的颜色
    for i in range(len(bond_no[0])):
        a1 = int(bond_no[0][i]) #表示第一个数组中的相互作用的原子索引
        a2 = int(bond_no[1][i]) #表示第二个数组中相互作用的原子索引
        
        bond_color = smi_att[a1,a2] ##得到键的权重
        ###归一化，颜色差别不明显
        #atom_weights = (bond_color-min_value) / (max_value - min_value)
        ###该用法里面需要二维数组
        #atom_weights = (2 * (bond_color - min_value) / (max_value - min_value)) - 1
        bond_atten.append(bond_color)  ###归一化后的权重
        bond = mol.GetBondBetweenAtoms(a1,a2).GetIdx() ###获取键的索引
        bond_list.append(bond)
        atom_list.append((a1,a2))
        
    
    norm=matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
    cmap = cm.get_cmap('coolwarm') #选择了颜色映射的颜色方案
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    bond_colors = {bond_list[i]: plt_colors.to_rgba(bond_atten[i]) for i in range(len(bond_list))}
    drawer = rdMolDraw2D.MolDraw2DCairo(500,500) #创建一个 MolDraw2DCairo 对象，指定图像的宽度和高度为 500x500 像素
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol, 
                             highlightBonds=bond_list,
                             highlightBondColors=bond_colors)
    ##将着色的分子结构绘制到绘图对象 drawer 上
    #使用 highlightBonds 参数高亮显示指定的化学键，highlightBondColors 参数传递了 bond_colors，以便将之前计算的颜色应用到相应的化学键上
    
    ##画色带
    fig,ax1 = plt.subplots()
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 
    cb = plt.colorbar(plt_colors,cax=cax) #创建一个颜色条（colorbar）
    cb.set_ticks([]) #将颜色条的刻度设置为空 
   
    #确定要保存的图像文件的名称
    output_name = str(smile)
    output_name = output_name.replace('/','%')
    output_name = output_name.replace('\\','%')
    #就截取前 50 个字符，以确保文件名不会太长
    if len(output_name) > 50:
        output_name = output_name[0:50]
    str1 = path + '/' + output_name + '.jpg' ##smiles可视化后保存的名字
    with open(str1, 'wb') as file: #以二进制写入模式（'wb'）打开文件
        file.write(drawer.GetDrawingText()) # 将绘制的图像数据写入文件
        print(f'Produce the interpretation molecule graph in {str1}')
    return atom_list,bond_atten

def interp_graph(args):
    print('Load args.')
    ###第一步导入超参数
    scaler = get_scaler(args.model_path) ##如果有，则获取均值或标准差，否则无
    train_args = load_args(args.model_path) ###已训练完毕的模型超参数

    ##检查args对象里面是否有相应的键和值，如没有则添加，以便保持一致性
    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    ###第二步导入数据
    print('Load data.')
    test_data = load_data(args.predict_path,args) ###是MoleDataset对象，里面有多个MoleData对象
    fir_data_len = len(test_data) ###
    all_data = test_data
    if fir_data_len == 0:
        raise ValueError('Data is empty.')

    ###检查数据是否有效
    smi_exist = []
    for i in range(fir_data_len):
        if test_data[i].mol is not None:
            smi_exist.append(i)
    test_data = MoleDataSet([test_data[i] for i in smi_exist]) ##有效的MoleData封装成新的MoleDatasets对象
    now_data_len = len(test_data)
    print('There are ',now_data_len,' smiles in total.')
    if fir_data_len - now_data_len > 0:
        print('There are ',fir_data_len - now_data_len, ' smiles invalid.')
    
    test_smile = test_data.smile() ##得到所有有效的MoleData的smiles
    test_label = test_data.label()

    ###第三步导入模型
    print('Load model')
    model = load_model(args.model_path,args.cuda,pred_args=args) ###先导入了模型超参数构建模型框架，然后导入模型参数，模型就固定了
    test_pred = predict(model,test_data,args.batch_size,scaler) ###得到属于每个标签的概率
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    print('True label: ',test_label)
    print('Pred label: ',test_pred)
    
    atts_out = get_atts_out() ###获取模型的注意力权重矩阵，有多个头就有多个注意力权重矩阵
    nhead = args.nheads + 1
    
    total_atten = []
    total_bond_list = []
    total_bond_atten = []
    for i in range(now_data_len):
        smile = test_smile[i]
        smi_att = atts_out[(i+1) * nhead - 1] ##一个SMILES存在args.nheads+1个权重矩阵（N*N），因为有两层GAT层，前一个是多头，后一个是单头
        ##我们最终取的权重矩阵是最后一个权重矩阵，即第nheads个权重矩阵，每个SMILES进行预测是就会有权重矩阵
        total_atten.append(smi_att)
        atom_list,bond_atten = drawmol_bond(smile,smi_att,args.figure_path) ###进行分子图的可视
        total_bond_list.append(atom_list)
        total_bond_atten.append(bond_atten)

    smile_df = pd.DataFrame(test_smile,columns=['SMILES'])
    true_label = pd.DataFrame(test_label,columns=['True_1A2','True_2C9','True_2C19','True_2D6','True_3A4'])
    pred_label = pd.DataFrame(test_pred,columns=['Pred_1A2','Pred_2C9','Pred_2C19','Pred_2D6','Pred_3A4'])
    total_bond_df = pd.DataFrame(total_bond_list)
    total_atten_df = pd.DataFrame(total_bond_atten)
    total = pd.concat([smile_df,true_label,pred_label,total_bond_df,total_atten_df],axis=1)
    total.to_csv(os.path.join(args.figure_path, 'visual_external.csv'))
    

if __name__ == '__main__':
    args = set_intergraph_argument()
    model_path = args.model_path
    model_dir = os.path.join(model_path, f"Seed_0")
    args.model_path = os.path.join(model_dir, f'0_model.pt')
    interp_graph(args)