# DeepModeling Hackathon 2021

## 活动介绍

此次由**中科大苏州高等研究院**与**北京深势科技有限公司**联合举办的**首届DeepModeling黑客马拉松大赛**，希望通过汇集一批年轻的科研工作者、技术工程师交流开发经验、提升开发能力，为未来提供高效、可复用、可持续优化的建模工具探索新的协同模式。无论你是技术大神还是萌新小白，无论你是科研工作者还是技术工程师，你都可以尝试单独成队或和你的团队聚在一起实现你们的创意！无论你来自什么院校，来自什么部门，来参加就是了！

## 活动安排
具体安排请关注“**深度势能**”公众号，并留意最新推送。

## 比赛日程
### 赛前准备阶段（8月15日之前）

**7月1日**：放出部分赛题，选手开始提交proposal及赛题意见<br>
**7月11日**：更新最终赛题并确定数据集<br>
**7月12日**：开始线上直播软件教学（具体直播安排见公众号推送）<br>
**7月25日**：报名截止并停止提交proposal，开始对接导师指导交流<br>
**(7月1日—8月15日是大家熟悉题目、开始动手研究题目的好时机哦)**<br>

### 活动开始（8月16日—20日，苏州）

**8月16日—17日**：报告分享与教学：
>聆听各位嘉宾的分享报告，学习使用相关软件工具，并随时交流答疑。

**8月18日—20日**：正式比赛：
>放出**bonus题目**，各组选定题目比赛，赛后专家团队评分，选出优秀队伍做展示并颁奖。<br>

每队**7月25日之前可以反复提交proposal**，于正式比赛当天确定1-2道题目，此部分题目的分数主要基于完成度和创新度；<br>
Bonus题目可选多道，不需要提前选定，可根据时间安排即答即交，我们根据此部分题目的完成度和题目数量非线性给分。

## 题目主题一览

### AI赛道：理解物理建模之智慧，洞若观火（编号A开头）

#### [A1.探究原子邻域描述子的可解释性](#a1-interpretability-of-descriptors-in-deepmd-kit)

#### [A2.统计并抽取网络信息，实现模型压缩](#A2-Analysis-of-information-quantity-and-Model-compression-in-DeepMD-kit)

#### [A3.利用网络结构搜索（NAS）寻找最优模型](#A3-Neural-Architecture-Search-(NAS)-in-energy/force-fitting)

#### ...

### 高性能赛道：打磨物理建模之利刃，吹毫立断（编号H开头）

#### [H1.DeepMD-kit混合精度训练加速](#H1-Mixed-precision-training-in-DeepMD-kit)

#### [H2.Abacus近邻原子搜索优化](#H2-Improve-of-neighbor-atoms-searching-code-in-Abacus)

#### [H3.FEALPy高效组装单元矩阵](#H3-Efficient-implementation-for-the-finite-element-local-assembly)

#### ...

### 科学计算：将借物理建模之羽翼，直飞云空（编号S开头）

#### [S1.DeepMD-kit实现热导/扩散系数/径向分布函数计算工作流](#S1-Workflows-for-computing-physical-properties)

#### [S2.Abacus实现材料能带计算工作流](#S2-Workflows-for-AbacusMaterials-Project-for-calculate-electric-bands)

#### [S3.FEALPy实现基于Bernstein多项式的有限元方法](#S3-Finite-element-space-based-on-the-Bernstein-polynomial)

#### ...

### 注：题目细节要求及数据集将在7月11日最终确定
<br>
<br>

## AI赛道：理解物理建模之智慧，洞若观火

### A1. Interpretability of descriptors in DeepMD-kit

#### 维度：AI+DeepMD-kit+科研创新

#### Background&Problem：

1.***Model interpretability*** is a crucial property for machine learning, which is also challenging for researchers.

2.Plenty of efforts have been made to develop Explainable Artificial Intelligence (XAI, see a review in [https://doi.org/10.1016/j.inffus.2019.12.012](https://doi.org/10.1016/j.inffus.2019.12.012) ), such as:
>ICE ([https://www.tandfonline.com/doi/full/10.1080/10618600.2014.907095](https://www.tandfonline.com/doi/full/10.1080/10618600.2014.907095) )

>LIME ([https://dl.acm.org/doi/10.1145/2939672.2939778](https://dl.acm.org/doi/10.1145/2939672.2939778) ),

>SHAP ([https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) ),

>RETAIN ([https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html](https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html) )

3.**Descriptors in DeepMD-kit** contain the environment information of each central atom, and are fed into fitting network to obtain the final output. The interpretability of these descriptors remains no more investigation, which is open to all kinds of responsible explanation.

#### Goal:

Investigate the post-hoc interpretability of descriptors in DeepMD-kit.

#### Possible solution:

1.Design a method or utilize methods mentioned above to illustrate the interpretability of descriptors in DeepMD-kit;

2.Possibly do some visualization;

3.**PaddlePaddle** provides various tools convenient for analysis and Interpretation:
>Interpretation [https://github.com/PaddlePaddle/InterpretDL](https://github.com/PaddlePaddle/InterpretDL)

In addition, **if you choose PaddlePaddle, mentorship can be provided.**
<br>
<br>
### A2. Analysis of information quantity and Model compression in DeepMD-kit

#### 维度：AI+DeepMD-kit+科研创新

#### Background&Problem:
1.***Model compression*** in AI is solution to efficiency and information distillation. 

2.Networks in DeepMD-kit may contain some redundant parameters.

3.The information quantity contained in different layers is not clearly compared and may be a way to do network pruning.

#### Goal: 
1.Make an analysis on the information quantity contained in different layers. 

2.Use some strategies to do model compression and make the network smaller and more efficient without losing too much accuracy.

#### Possible solution:
1.Use mutual information to measure the information quantity contained in different layers; 

2.Use quantization、knowledge distillation、low-rank factorization or PCA and so on to do model compression; 

3.**PaddlePaddle** provides various tools convenient for analysis and model compression:
>Interpretation https://github.com/PaddlePaddle/InterpretDL 

>Model compression https://github.com/PaddlePaddle/PaddleSlim 

In addition, **if you choose PaddlePaddle, mentorship can be provided.**
<br>
<br>
### A3. Neural Architecture Search (NAS) in energy/force fitting

#### 维度：AI+DeepMD-kit+科研创新

#### Background&Problem：
1.***Neural Architecture Search（NAS)*** may be solution to tedious network architecture design（https://arxiv.org/abs/1707.07012), which can learn the model architectures directly on the dataset of interest.

2.Architectures in DeepMD-kit are delicately designed based on linear and resnet blocks with theoretical hypothesis，while we are open to any other available architecture design.
#### Goal:  
1.Use NAS(or meta learning) to automatically choose parameters for better performance in energy & force fitting;

2.Use NAS to search for a better network, which may outperform standard DeepMD-kit. 
#### Possible solution： 
1.For **Goal2**, in energy/force fitting (or you can simultaneously do both), use NAS to search for the best architectural building block on a small dataset and then transfer the best block architecture to a larger dataset, then train a new model. Finally, compare with standard trained DeepMD-kit on validation dataset.

2.**PaddlePaddle** provides various tools convenient for NAS:
>Slim https://github.com/PaddlePaddle/PaddleSlim

>NAS https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.1.0/docs/zh_cn/quick_start/static/nas_tutorial.md 

In addition, **if you choose PaddlePaddle, mentorship can be provided.**
<br>
<br>

## 高性能赛道：打磨物理建模之利刃，吹毫立断
### H1. Mixed precision training in DeepMD-kit
#### 维度：高性能+DeepMD-kit+科研创新
#### Background&Problem：
1.**Mixed precision training** is widely used in HPC. Under the premise of ensuring the output accuracy within a certain range, the training process can be accelerated by using **single precision** or **semi-precision**.

2.**DeepMD-kit** adopts double precision training by default and has a single precision training interface, but there is no corresponding exploration for semi-precision training.

#### Goal: 
With specific device and specific Tensorflow version, try to use the mixture of single precision and semi-precision in the DeepMD-Kit training process, and speed up the training process under the premise of ensuring the output precision(**detailed requirements will be given later**).
#### Possible solution：
1.You can feel free to change the float precision during training process;

2.To achieve the best performance, you can also change the neuron parameters to refine the number of network layers;

3.***Note***：To ensure the accuracy of semi-precision training, attention should be paid to gradient explosion and gradient vanishing. In addition, be careful about the matrix dimension in semi-precision Tensor Core.
<br>
<br>
### H2. Improve of neighbor atoms searching code in Abacus
#### 维度：高性能+Abacus+工程开发
#### Background&Problem：
1.DFT software calculation needs **truncation with radius cutoff**, efficient searching and fast traversing of other atoms around each atom is an important problem to improve the computational speed.

2.The existing neighbor searching code in **Abacus** adopts plenty of bit manipulations and is very efficient even up to millions of atoms.

3.The existing neighbor searching code has many steps and we don’t know if the code of each step is efficient enough. 
#### Goal: 
1.Try to understand the existing code of searching neighbor atoms.(**write notes to show your comments**)

2.Try to **remove the BOOST math library**, and test efficiency of each step for target examples.

3.Try to **improve the efficiency** of codes.
#### Possible solution：
1.To achieve the best performance, you can **feel free to use MPI or openMP or CUDA** to accelerate.

2.***Note***：you’d better don’t use code beyond c++11.
<br>
<br>

### H3. Efficient implementation for the finite element local assembly
#### 维度：高性能+FEALPy+工程开发
#### Background&Problem:
1.**Assembling the element matrix** is a key step in the finite element method. To get the element matrices, one need to calculate the numerical integral of the product between any two local basis functions on each mesh element.

2.At present, **FEALPy** uses the einsum function in Numpy to get the element matrices,but this function does not support multi-core calculation. As a result, FEALPy can not make full use of the computer’s  multi-core computing resources in this step.
#### Goal:
Design and develop a special multi-core version function to replace the einsum function for FEALPy.
#### Possible solution
One can find some detailed discussion in the following paper:

>Luporini F, Varbanescu A L, Rathgeber F, et al. Cross-loop optimization of arithmetic intensity for finite element local assembly[J]. ACM Transactions on Architecture and Code Optimization (TACO), 2015, 11(4): 1-25.
<br>

## 科学计算：将借物理建模之羽翼，直飞云空
### S1. Workflows for computing physical properties
#### 维度：科学计算+DeepMD-kit+科研创新
#### Background&Problem:
1.**Well-designed workflows** are important for the transparency and reproducibility of scientific computing tasks. In addition, they are very useful for both pedagogical and production purposes.

2.Practitioners in scientific computing typically lack trainings for managing and maintaining workflows.

3.Here we list a few tasks for which we pay particular attention to the workflow perspective. One may propose their own workflows（choose one or more of the followings）: 
1) to compute the **heat conductance of water** using a Deep Potential model; 

2) to compute the **radial distribution functions** using a Deep Potential model; 

3) to compute the **diffusion coefficient** using a Deep Potential model.
#### Goal: 
Develop good workflows for large-scale and computationally-intensive tasks, which would help to boost the efficiency of scientific computation jobs.
#### Possible solution：
Design and develop a workflow using Apache airflow or aiida, or other workflow management tools. One may take dpti (https://github.com/deepmodeling/dpti ) as an example. 
<br>
<br>

### S2. Workflows for Abacus+Materials Project for calculate electric bands
#### 维度：科学计算+Abacus+工程开发
#### Background&Problem:
1.**Abacus** is accurate and effective DFT software to calculate material properties.

2.It is important to automatically and quickly compute the examples in the open material database.
#### Goal: 
Developing a workflow to calculate **electric bands** with Abacus for at least 100 structures downloaded in **Materials Project**. (**Compounds of III-V families elements and II-VI families elements is recommended.**)
#### Possible solution：
1.The input files example and key parameter setting would be listed.

2.You can find Materials Project data in https://www.materialsproject.org/, and there is official API to get data from this database.

3.You can find some workflows examples in https://gitee.com/ShenZhenXiong/AseAbacusInterface/blob/master/AseAbacusV20200227/example/highflux/ase_abacus_highflux.py .
<br>
<br>

### S3. Finite element space based on the Bernstein polynomial
#### 维度：科学计算+FEALPy+工程开发
#### Background&Problem:
1.In **FEALPy**, the basis functions of **Lagrange finite element space** defined on the simplex (interval, triangle or tetrahedron) meshes is constructed based on the **barycentric coordinates**, which does not need to introduce the reference element. Furthermore, they satisfy the interpolation property, that is, each basis function takes 1 at one of interpolation points and 0 at the other on each element.

2.The calculation of higher derivatives and numerical integrals of this kind of basis function is cumbersome.

3.If the interpolation property is not required, one can use the **Bernstein polynomial** based on barycentric coordinates,which owns simpler form and has clear formulas for derivation and integral calculation.

#### Goal: 
1.Design and develop a space class named `BernsteinFiniteElementSpace`, which should have same interfaces as the `LagrangeFiniteElementSpace` class in FEALPy;

2.All function should be implemented by the **array-oriented** and **dimension independent** techniques;

3.Verify the correctness of this space class by **solving Poisson equation**.
#### Possible solution：
1.One can find the definition of Bernstein polynomial in [Wiki](https://en.wikipedia.org/wiki/Bernstein_polynomial). 

2.One can find the Bernstein basis based on barycentric coordinates in the follow paper:
>Feng L, Alliez P, Busé L, et al. Curved optimal delaunay triangulation[J]. ACM Transactions on Graphics, 2018, 37(4): 16.

### 赛题正在持续更新中
## 其他信息
### 报名问卷:
[点此填写](https://www.wjx.top/vj/toN7B3a.aspx)
### 提交proposal:
[点此填写](https://www.wjx.top/vj/wbKckRf.aspx)
