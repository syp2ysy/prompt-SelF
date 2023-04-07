<div align="center">
<h1>SegGPT: Segmenting Everything In Context </h1>

[Xinlong Wang](https://www.xloong.wang/)<sup>1*</sup>, &nbsp; [Xiaosong Zhang](https://scholar.google.com/citations?user=98exn6wAAAAJ&hl=en)<sup>1*</sup>, &nbsp; [Yue Cao](http://yue-cao.me/)<sup>1*</sup>, &nbsp; [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>2</sup>, &nbsp;  [Chunhua Shen](https://cshen.github.io/)<sup>2</sup>, &nbsp; [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), &nbsp; <sup>2</sup>[ZJU](https://www.zju.edu.cn/english/), &nbsp; <sup>3</sup>[PKU](https://english.pku.edu.cn/)

Enjoy the [Demo](https://huggingface.co/spaces/BAAI/SegGPT)


<br>
  
<image src="reason.png" width="720px" />
<br>

</div>

<br>
  The In-Context Learning (ICL) is to understand a new task via a few demonstrations (aka. prompt) and predict new inputs without tuning the models. While it has been widely studied in NLP, it is still a relatively new area of research in computer vision. To reveal the factors influencing the performance of visual in-context learning, this paper shows that prompt selection and prompt fusion are two major factors that have a direct impact on the inference performance of visual context learning. Prompt selection is the process of identifying the most appropriate prompt or example to help the model understand new tasks. This is important because providing the model with relevant prompts can help it learn more effectively and efficiently. Prompt fusion involves combining knowledge from different positions within the large-scale visual model. By doing this, the model can leverage the diverse knowledge stored in different parts of the model to improve its performance on new tasks. Based these findings, we propose a simple framework prompt-SelF for visual in-context learning. Specifically, we first use the pixel-level retrieval method to select a suitable prompt, and then use different prompt fusion methods to activate all the knowledge stored in the large-scale model, and finally ensemble the prediction results obtained from different prompt fusion methods to obtain the final prediction results. And we conduct extensive experiments on single-object segmentation and detection tasks to demonstrate the effectiveness of prompt-SelF. Remarkably, the prompt-SelF has outperformed OSLSM based meta-learning in 1-shot segmentation for the first time. This indicated the great potential of visual in-context learning.

[[Paper]]()



## Citation


## Contact

**
