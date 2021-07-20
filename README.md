# DeepModeling Hackathon 2021

## 活动介绍

此次由**中科大苏州高等研究院**与**北京深势科技有限公司**联合举办的**首届DeepModeling黑客马拉松大赛**，希望通过汇集一批年轻的科研工作者、技术工程师交流开发经验、提升开发能力，为未来提供高效、可复用、可持续优化的建模工具探索新的协同模式。无论你是技术大神还是萌新小白，无论你是科研工作者还是技术工程师，你都可以尝试单独成队或和你的团队聚在一起实现你们的创意！无论你来自什么院校，来自什么部门，来参加就是了！

## 活动安排
具体安排请关注“**深度势能**”公众号，并留意最新推送。

## 软件教学直播日程
**7月19日** ABACUS使用指导与赛题介绍<br>
**7月21日** DeePMD-kit使用指导与赛题介绍<br>
**7月23日** NVIDIA线下活动与直播教程大礼包（含NVIDIA、DeePMD-kit、PaddlePaddle关于科学计算中的高性能优化分享）<br>
**7月26日** FEALPy使用指导与赛题介绍<br>
**7月28日** PaddlePaddle使用指导与赛题介绍<br>

## 直播录屏&PPT
**点击下载**: [code:683b](https://pan.baidu.com/s/16EAp-jBqO6pB2VysB-IVNA)

## 比赛日程
### 赛前准备阶段（8月15日之前）

**7月1日**：放出部分赛题，选手开始提交proposal及赛题意见<br>
**7月11日**：更新最终赛题并确定数据集<br>
**7月12日**：开始线上直播软件教学（具体直播安排见公众号推送）<br>
**7月30日**：报名截止并停止提交proposal，开始对接导师指导交流<br>
**(7月1日—8月15日是大家熟悉题目、开始动手研究题目的好时机哦)**<br>

### 活动开始（8月16日—20日，苏州）

**8月16日—17日**：报告分享与教学：
>聆听各位嘉宾的分享报告，学习使用相关软件工具，并随时交流答疑。

**8月18日—20日**：正式比赛：
>放出**bonus题目**，各组选定题目比赛，赛后专家团队评分，选出优秀队伍做展示并颁奖。<br>

每队**7月30日之前可以反复提交proposal**，于正式比赛当天确定1-2道题目，此部分题目的分数主要基于完成度和创新度；<br>
Bonus题目可选多道，不需要提前选定，可根据时间安排即答即交，我们根据此部分题目的完成度和题目数量非线性给分。

## 题目主题一览

### AI赛道：理解物理建模之智慧，洞若观火（编号A开头）

#### [A1.探究原子邻域描述子的可解释性](#a1-interpretability-of-descriptors-in-deepmd-kit)

#### [A2.统计并抽取网络信息，实现模型压缩](#A2-Analysis-of-information-quantity-and-Model-compression-in-DeePMD-kit)

#### [A3.利用网络结构搜索（NAS）寻找最优模型](#A3-Neural-Architecture-Search-in-energyforce-fitting)

### 高性能赛道：打磨物理建模之利刃，吹毫立断（编号H开头）

#### [H1.DeePMD-kit混合精度训练加速](#H1-Mixed-precision-training-in-DeePMD-kit)

#### [H2.ABACUS近邻原子搜索优化](#H2-Improve-the-algorithms-in-searching-for-neighboring-atoms)

#### [H3.FEALPy高效组装单元矩阵](#H3-Efficient-implementation-for-the-finite-element-local-assembly)

### 科学计算：将借物理建模之羽翼，直飞云空（编号S开头）

#### [S1.DeePMD-kit实现热导/扩散系数/径向分布函数计算工作流](#S1-Workflows-for-computing-physical-properties)

#### [S2.ABACUS实现材料能带计算工作流](#S2-Workflows-for-ABACUSMaterials-Project-for-calculating-band-gaps)

#### [S3.FEALPy实现基于Bernstein多项式的有限元方法](#S3-Finite-element-space-based-on-the-Bernstein-polynomial)


### 注：题目细节要求及数据集已确定，后续可能只会有细微调整，如有问题欢迎在群里反馈
<br>
<br>

## AI赛道：理解物理建模之智慧，洞若观火

### A1. Interpretability of descriptors in DeePMD-kit

#### 维度：AI+DeePMD-kit+科研创新

#### Background&Problem：

1.***Model interpretability*** is a crucial property for machine learning, which is also challenging for researchers.

2.Plenty of efforts have been made to develop Explainable Artificial Intelligence (XAI, see a review in [https://doi.org/10.1016/j.inffus.2019.12.012](https://doi.org/10.1016/j.inffus.2019.12.012) ), such as:
>ICE ([https://www.tandfonline.com/doi/full/10.1080/10618600.2014.907095](https://www.tandfonline.com/doi/full/10.1080/10618600.2014.907095) )

>LIME ([https://dl.acm.org/doi/10.1145/2939672.2939778](https://dl.acm.org/doi/10.1145/2939672.2939778) ),

>SHAP ([https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) ),

>RETAIN ([https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html](https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html) )

3.**Descriptors in DeePMD-kit** contain the environment information of each central atom, and are fed into fitting network to obtain the final output. The interpretability of these descriptors remains no more investigation, which is open to all kinds of responsible explanation.

#### Goal:

1.Investigate the post-hoc interpretability of descriptors in copper systems of different crystal structures (i.e. descriptors of `Cu` in different crystal structures) in DeePMD-kit.

2.Requirements: 

(1)set `descriptor` as `se_a` and train **100w steps** on 6 copper systems as a whole training set, then make analysis for **6 given systems** (it will take hours to train, so it’s better to start up early);

(2)add `type_one_side` = `Fasle` of descriptor parameters in input.json, an example is provided here: [input.json](A1/input.json), feel free to change other training parameters.

#### Possible solution:

1.This is a problem for people who prefer to do some analysis and may not require much change to the code and implementation.

2.To get the basic score, you need to be familiar with the `se_a` descriptor in deepmd-kit and the reasons for its construction. Design a method or utilize methods mentioned above to illustrate the interpretability of descriptors in DeePMD-kit, possibly do some clustering or visualization;

3.Feel free to do other relevant analysis and get the bonus score.

4.**PaddlePaddle** provides various tools convenient for analysis and Interpretation:
>Interpretation [https://github.com/PaddlePaddle/InterpretDL](https://github.com/PaddlePaddle/InterpretDL)

In addition, **if you choose PaddlePaddle, mentorship can be provided.**

#### Get Start: 
You can see [here](https://github.com/deepmodeling/deepmd-kit) for coding instruction, and `develop` branch is where you accomplish this project. You might be able to get the descriptors mainly around `deepmd-kit/deepmd/descriptor` in `develop` branch. Be aware of **online tutorial of DeePMD-kit** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose DeePMD-kit relevant projects. 

#### Dataset&Materials:
1.Copper systems in different crystal structures: [code:4m44](https://pan.baidu.com/s/19CBKaN-ZEidWc3mChAYJlg).

#### Scoring point:
1.The rationality and completeness of the analysis.

2.The innovativeness of the analysis.

#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment and the analysis results.
<br>
<br>
### A2. Analysis of information quantity and Model compression in DeePMD-kit

#### 维度：AI+DeePMD-kit+科研创新

#### Background&Problem:
1.***Model compression*** in AI is solution to efficiency and information distillation. 

2.Networks in DeePMD-kit may contain some redundant parameters.

3.The information quantity contained in different layers is not clearly compared and may be a way to do network pruning.

#### Goal: 
1.Train the DeePMD-kit **100w steps** on the given system below and freeze the model (feel free to set other parameters in input.json);

2.Make an analysis on the information quantity contained in **different layers**;

3.Use some strategies to do model compression and make the network smaller and more efficient without losing too much accuracy.

#### Possible solution:
1.To get the basic score, you need to have a clear understanding of the overall model structure of DeePMD-kit.
 
2.You can use mutual information to measure the information quantity contained in different layers; 

3.You can use quantization、knowledge distillation、low-rank factorization or PCA and so on to do model compression; 

4.**PaddlePaddle** provides various tools convenient for analysis and model compression:
>Interpretation https://github.com/PaddlePaddle/InterpretDL 

>Model compression https://github.com/PaddlePaddle/PaddleSlim 

In addition, **if you choose PaddlePaddle, mentorship can be provided.**

#### Get Start: 
You can see [here](https://github.com/deepmodeling/deepmd-kit) for coding instruction, and `develop` branch is where you accomplish this project. You might be able to get the descriptors mainly around `deepmd-kit/deepmd/descriptor` in `develop` branch. Be aware of **online tutorial of DeePMD-kit** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose DeePMD-kit relevant projects. 

#### Dataset&Materials:
1.Small dataset: water example contained in `example/water/data`, you can also download from [here](https://github.com/deepmodeling/deepmd-kit/tree/devel/examples/water/data);

2.Challenge dataset: Cu system: [code:ifvk](https://pan.baidu.com/s/1QBP1CyshrnBV8UiKGQnSGw) (which is one of the six systems of dataset in [A1](#a1-interpretability-of-descriptors-in-deepmd-kit)).

#### Scoring point:
1.The rationality and completeness of the analysis.

2.The effectiveness and efficiency of the compression (the amount of parameters in compressed model, the inference speed and so on, compared with the standard model).

#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment and the analysis results.

2.a copy of code that can run directly (a trained model included).
<br>
<br>
### A3. Neural Architecture Search in energy/force fitting

#### 维度：AI+DeePMD-kit+科研创新

#### Background&Problem：
1.***Neural Architecture Search（NAS)*** may be solution to tedious network architecture design（https://arxiv.org/abs/1707.07012), which can learn the model architectures directly on the dataset of interest.

2.Architectures in DeePMD-kit are delicately designed based on linear and resnet blocks with theoretical hypothesis，while we are open to any other available architecture design.
#### Goal:  
1.Use NAS(or meta learning) to automatically change parameters (`neuron`, `lr` and `pref` are recommended) in input.json for better performance in energy & force fitting, after **20w steps** training on the data systems given below;

2.Use NAS to search for a better network, which may outperform standard DeePMD-kit, after **100w steps** training on the data systems given below. 
#### Possible solution： 
1.This problem is the one with high freedom, you may use any tools to achieve the goals.

2.For **Goal2**, in energy/force fitting (or you can simultaneously do both), you can use NAS to search for the best architectural building block on a small part of dataset and then transfer the best block architecture to a larger part, then train a new model. Finally, compare with standard trained DeePMD-kit on validation part.

3.**PaddlePaddle** provides various tools convenient for NAS:
>Slim https://github.com/PaddlePaddle/PaddleSlim

>NAS https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.1.0/docs/zh_cn/quick_start/static/nas_tutorial.md 

In addition, **if you choose PaddlePaddle, mentorship can be provided.**
#### Get Start: 
You can see [here](https://github.com/deepmodeling/deepmd-kit) for coding instruction, and `develop` branch is where you accomplish this project. You might be able to get the descriptors mainly around `deepmd-kit/deepmd/descriptor` in `develop` branch. Be aware of **online tutorial of DeePMD-kit** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose DeePMD-kit relevant projects. 

#### Dataset&Materials:
1.Small dataset: water example contained in `example/water/data`, you can also download from [here](https://github.com/deepmodeling/deepmd-kit/tree/devel/examples/water/data);

2.Challenge dataset: Cu system: [code:ifvk](https://pan.baidu.com/s/1QBP1CyshrnBV8UiKGQnSGw) (which is one of the six systems of dataset in [A1](#a1-interpretability-of-descriptors-in-deepmd-kit)).

#### Scoring point:
1.The effectiveness and correctness of the NAS procedure;

2.The innovativeness of the NAS procedure.

#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly (a trained model included).
<br>
<br>

## 高性能赛道：打磨物理建模之利刃，吹毫立断
### H1. Mixed precision training in DeePMD-kit
#### 维度：高性能+DeePMD-kit+科研创新
#### Background&Problem：
1.**Mixed precision training** is widely used in HPC. Under the premise of ensuring the output accuracy within a certain range, the training process can be accelerated by using **single precision** or **semi-precision**.

2.**DeePMD-kit** adopts double precision training by default and has a single precision training interface, but there is no corresponding exploration for semi-precision training.

#### Goal: 
1.With single v100 and specific Tensorflow version (will be given later), try to use the mixture of single precision and semi-precision in the DeePMD-Kit training process, and speed up the training process under the premise of ensuring the output precision (threshold of error will be given later).

2.After the same time training process (you can set the time), compare the validate loss with the standard training procedure.
#### Possible solution：
1.You can feel free to change the float precision during training process;

2.To achieve the best performance, you can also change the neuron parameters to refine the number of network layers;

3.***Note***：To ensure the accuracy of semi-precision training, attention should be paid to gradient explosion and gradient vanishing. In addition, be careful about the matrix dimension in semi-precision Tensor Core.
#### Get Start: 
You can see [here](https://github.com/deepmodeling/deepmd-kit) for coding instruction, and `develop` branch is where you accomplish this project. You might be able to get the descriptors mainly around `deepmd-kit/deepmd/descriptor` in `develop` branch. Be aware of **online tutorial of DeePMD-kit** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose DeePMD-kit relevant projects. 

#### Dataset&Materials:
1.Small dataset: water example contained in `example/water/data`, you can also download from [here](https://github.com/deepmodeling/deepmd-kit/tree/devel/examples/water/data);

2.Challenge dataset: Cu system: [code:ifvk](https://pan.baidu.com/s/1QBP1CyshrnBV8UiKGQnSGw) (which is one of the six systems of dataset in [A1](#a1-interpretability-of-descriptors-in-deepmd-kit)).

#### Scoring point:
1.The correctness and efficiency of the training boosting ;

2.The innovativeness of the procedure.

#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly (a trained model included).
<br>
<br>

### H2. Improve the algorithms in searching for neighboring atoms 
#### 维度：高性能+ABACUS+工程开发
#### Background&Problem：
1.For **large-scale DFT calculations**, **ABACUS** adopts atomic orbitals with **strict radius cutoffs**. Efficient searching and recording neighboring atoms is an essential procedure that has been implemented in ABACUS. The current algorithms in searching neighbors can be used for large systems, but **still has potentials to be further improved**.

#### Goal: 
1.Read and understand the algorithms for searching neighboring atoms.

2.Remove the dependence of the **Hash library** in **BOOST math library** by implementing your own algorithm or using the STL in C++. Ensure the new algorithm can work correctly and compare the efficiency of the new and old algorithms. 

3.Try to improve the efficiency of searching neighbors.
#### Possible solution：
1.Use **MPI, openMP or CUDA** to accelerate the efficiency of the code.

#### Get Start: 
You can see [here](https://github.com/deepmodeling/abacus-develop/tree/reconstruction) for coding instruction, and `reconstruction` branch is where you accomplish this project. The code to be edited might mainly lies in `abacus-develop/source/module_neighbor` in `reconstruction` branch. Be aware of **online tutorial of ABACUS** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose ABACUS relevant projects. 

#### Dataset&Materials:
1.Small dataset: example dataset in code package: `abacus-develop/tests/501_NO_neighboring_GaAs512/`, you can also download here: [501_NO_neighboring_GaAs512.zip](H2/501_NO_neighboring_GaAs512.zip).

2.Challenge dataset: C262144-neighbor, download here: [C262144-neighbor.zip](H2/C262144-neighbor.zip).

#### Scoring point:
1.The correctness of the implementation;

2.The efficiency of the code compared with the standard ABACUS procedure.


#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly.
<br>
<br>

### H3. Efficient implementation for the finite element local assembly
#### 维度：高性能+FEALPy+工程开发
#### Background&Problem:
1.**Assembling the element matrix** is a key step in the finite element method. To get the element matrices, one need to calculate the numerical integral of the product between any two local basis functions on each mesh element.

2.At present, **FEALPy** uses the einsum function in Numpy to get the element matrices,but this function does not support multi-core calculation. As a result, FEALPy can not make full use of the computer’s  multi-core computing resources in this step.
#### Goal:
Please see the code in [construct_stiff_matrix.py](H3/construct_stiff_matrix.py), design and develop a special **multi-core version** function to replace the **einsum function**.
#### Possible solution
One can find some detailed discussion in the following paper:

>Luporini F, Varbanescu A L, Rathgeber F, et al. Cross-loop optimization of arithmetic intensity for finite element local assembly[J]. ACM Transactions on Architecture and Code Optimization (TACO), 2015, 11(4): 1-25.

#### Get Start: 
Please see the [install.md](H3/install.md) file for installation FEALPy on your system. See [here](https://github.com/deepmodeling/fealpy) for coding instruction, and `master` branch is where you accomplish this project. Be aware of **online tutorial of FEALPy** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose FEALPy relevant projects.

#### Scoring point:
1.The correctness of the implementation;

2.The efficiency of the code compared with the standard single-core FEALPy procedure.


#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly.

<br>

## 科学计算：将借物理建模之羽翼，直飞云空
### S1. Workflows for computing physical properties
#### 维度：科学计算+DeePMD-kit+科研创新
#### Background&Problem:
1.**Well-designed workflows** are important for the transparency and reproducibility of scientific computing tasks. In addition, they are very useful for both pedagogical and production purposes.

2.Practitioners in scientific computing typically lack trainings for managing and maintaining workflows.

3.Here we list a few tasks for which we pay particular attention to the workflow perspective. One may propose their own workflows（choose one or more of the followings, **or you can inform us and put forward other workflows**）: 
1) to compute the **heat conductance of water** using a Deep Potential model; 

2) to compute the **radial distribution functions** using a Deep Potential model; 

3) to compute the **diffusion coefficient** using a Deep Potential model.
#### Goal: 
Develop good workflows for large-scale and computationally-intensive tasks, which would help to boost the efficiency of scientific computation jobs.
#### Possible solution：
Design and develop a workflow using Apache airflow or aiida, or other workflow management tools. One may take dpti (https://github.com/deepmodeling/dpti ) as an example. 
#### Get Start: 
You can see [here](https://github.com/deepmodeling/deepmd-kit) for coding instruction, and `develop` branch is where you accomplish this project. You might be able to get the descriptors mainly around `deepmd-kit/deepmd/descriptor` in `develop` branch. Be aware of **online tutorial of DeePMD-kit** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose DeePMD-kit relevant projects. 

#### Scoring point:
1.The correctness of the implementation;

2.The universality and transferability of the designed workflow.


#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly.
<br>
<br>

### S2. Workflows for ABACUS+Materials Project for calculating band gaps
#### 维度：科学计算+ABACUS+工程开发
#### Background&Problem:
1.**ABACUS** is a density functional theory software based on quantum mechanics, it can be used to predict properties of materials. It is a powerful tool to combine ABACUS with open database for materials, such as **Materials Project**.

#### Goal: 
1.Developing a workflow to calculate **band gaps** of materials with ABACUS for **at least 100 structures** downloaded from the Materials Project. **(Compounds of III-V family elements and II-VI family elements are recommended.)**

2.Make sure that the workflow code has good **universality** for other materials.

#### Get started: 
You can see [here](https://github.com/deepmodeling/abacus-develop/tree/reconstruction) for coding instruction, and `reconstruction` branch is where you accomplish this project. Be aware of **online tutorial of ABACUS** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose ABACUS relevant projects. 

#### Dataset&Material:
1.**An example** including input files and explanations for key parameters is provided: [example-Si-band.zip](S2/example-Si-band.zip) , which contains the first scf step and the second nscf step of Si-diamond band calculation.

2.The atomic structures and related information can be downloaded from the [Materials Project](https://www.materialsproject.org/), and there is an official API to get data from this database.

3.You can find similar workflow examples on [this website](https://gitee.com/ShenZhenXiong/AseAbacusInterface/blob/master/AseAbacusV20200227/example/highflux/ase_abacus_highflux.py).

4.You can download pseudopotential files [here](http://abacus.ustc.edu.cn/uploadfile/Libs/SG15_v1.0_Pseudopotential.zip).

5.The goal 100 material examples are listed in [Material_IDs.txt](/S2/Material_IDs.txt), which you need to download from [Materials Project](https://www.materialsproject.org/) and compute with ABACUS.

#### Scoring point:
1.The correctness of the implementation;

2.The universality and transferability of the designed workflow.


#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly.
<br>
<br>
### S3. Finite element space based on the Bernstein polynomial
#### 维度：科学计算+FEALPy+工程开发
#### Background&Problem:
1.In **FEALPy**, the basis functions of **Lagrange finite element space** defined on the simplex (interval, triangle or tetrahedron) meshes is constructed based on the **barycentric coordinates**, which does not need to introduce the reference element. Furthermore, they satisfy the interpolation property, that is, each basis function takes 1 at one of interpolation points and 0 at the other on each element.

2.The calculation of higher derivatives and numerical integrals of this kind of basis function is cumbersome.

3.If the interpolation property is not required, one can use the **Bernstein polynomial** based on barycentric coordinates, which owns simpler form and has clear formulas for derivation and integral calculation.

#### Goal: 
1.Design and develop a space class named `BernsteinFiniteElementSpace`, which should have same interfaces as the `LagrangeFiniteElementSpace` class in FEALPy;

2.All function should be implemented by the **array-oriented** and **dimension independent** techniques;

3.Verify the correctness of this space class by **solving Poisson equation**.
#### Possible solution：
1.One can find the definition of Bernstein polynomial in [Wiki](https://en.wikipedia.org/wiki/Bernstein_polynomial). 

2.One can find the Bernstein basis based on barycentric coordinates in the follow paper:
>Feng L, Alliez P, Busé L, et al. Curved optimal delaunay triangulation[J]. ACM Transactions on Graphics, 2018, 37(4): 16.

3.Please see [here](S3/) for the specific code implementation requirements.

#### Get Start: 
Please see the [install.md](H3/install.md) file for installation FEALPy on your system. See [here](https://github.com/deepmodeling/fealpy) for coding instruction, and `master` branch is where you accomplish this project. Be aware of **online tutorial of FEALPy** coming soon, and we will inform you in the **wechat group** one week before the talk, which is important for those who choose FEALPy relevant projects.

#### Scoring point:
1.The correctness of the implementation;


#### Submit:
A zip file which contains: 

1.a report detailing the process of the experiment;

2.a copy of code that can run directly.

## 其他信息
### 报名问卷:
[点此填写](https://www.wjx.top/vj/toN7B3a.aspx)
### 提交proposal:
[点此填写](https://www.wjx.top/vj/wbKckRf.aspx)
