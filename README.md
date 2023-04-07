<div align="center">
<h1>Exploring Effective Factors for Improving Visual In-Context Learning </h1>

Yanpeng Sun<sup>1,2</sup>, &nbsp; Qiang Chen<sup>1*</sup>, &nbsp; Jian Wang<sup>1</sup>, &nbsp; Jingdong Wang<sup>1</sup>, &nbsp; Zechao Li<sup>2</sup>

<sup>1</sup>Baidu VIS, &nbsp; <sup>2</sup>Nanjing University of Science and Technology


<br>
  
<image src="prompt-self_motivation.jpg" width="720px" />
<br>

</div>

<br>
  The In-Context Learning (ICL) is to understand a new task via a few demonstrations (aka. prompt) and predict new inputs without tuning the models. While it has been widely studied in NLP, it is still a relatively new area of research in computer vision. To reveal the factors influencing the performance of visual in-context learning, this paper shows that prompt selection and prompt fusion are two major factors that have a direct impact on the inference performance of visual context learning. Prompt selection is the process of identifying the most appropriate prompt or example to help the model understand new tasks. This is important because providing the model with relevant prompts can help it learn more effectively and efficiently. Prompt fusion involves combining knowledge from different positions within the large-scale visual model. By doing this, the model can leverage the diverse knowledge stored in different parts of the model to improve its performance on new tasks. Based these findings, we propose a simple framework prompt-SelF for visual in-context learning. Specifically, we first use the pixel-level retrieval method to select a suitable prompt, and then use different prompt fusion methods to activate all the knowledge stored in the large-scale model, and finally ensemble the prediction results obtained from different prompt fusion methods to obtain the final prediction results. And we conduct extensive experiments on single-object segmentation and detection tasks to demonstrate the effectiveness of prompt-SelF. Remarkably, the prompt-SelF has outperformed OSLSM based meta-learning in 1-shot segmentation for the first time. This indicated the great potential of visual in-context learning.

[[Paper]]()



## Citation


## Contact

**
