# Enhancing Transformer-Based Time Series Forecasting with Mixture of Experts.

Master's thesis repository: Enhancing Transformer-Based Time Series Forecasting with Mixture of Experts.

This repository is a fork of the [OpenLTM](https://github.com/thuml/OpenLTM) repository, which is used as a basis for the development of the work.

[Here](#openltm) you can see the original README.

## Description

## Usage

## OpenLTM

OpenLTM is a open codebase aiming to provide a pipeline to develop and evaluate large time-series models.

> For deep time series models and task-specific benchmarks, we recommend [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and this comprehensive [Survey](https://arxiv.org/abs/2407.13278).

:triangular_flag_on_post: **News** (2025.04) Many thanks for the implementation of [TTMs](https://arxiv.org/pdf/2401.03955) and other LLM4TS methods from [frndtls](https://github.com/frndtls).

:triangular_flag_on_post: **News** (2024.12) Many thanks for the implementation of [GPT4TS](https://arxiv.org/abs/2302.11939) from [khairulislam](https://github.com/khairulislam).

:triangular_flag_on_post: **News** (2024.10) We include several large time-series models, release pre-training code, and provide scripts.

### What is LTM

LTM (**L**arge **T**ime-Series **M**odel) is a series of scalable deep models built on foundation backbones (e.g. Transformers) and large-scale pre-training, which will be applied to a variety of time series data and diverse downstream tasks. For more information, here we list some related slides: [[CN]](https://cloud.tsinghua.edu.cn/f/1f3fdcf3304c4a82bc13/), [[Eng]](https://cloud.tsinghua.edu.cn/f/8a585e37f45f46fd97d0/).


### Model Checklist

- [x] **Timer-XL** - Timer-XL: Long-Context Transformer for Unified Time Series Forecasting. [[ICLR 2025]](https://arxiv.org/abs/2410.04803), [[Code]](https://github.com/thuml/Timer-XL)
- [x] **Moirai** - Unified Training of Universal Time Series Forecasting Transformers. [[ICML 2024]](https://arxiv.org/abs/2402.02592), [[Code]](https://github.com/SalesforceAIResearch/uni2ts)
- [x] **Timer** - Timer: Generative Pre-trained Transformers Are Large Time Series Models. [[ICML 2024]](https://arxiv.org/abs/2402.02368), [[Code]](https://github.com/thuml/Large-Time-Series-Model)
- [x] **Moment** - MOMENT: A Family of Open Time-series Foundation Model. [[ICML 2024]](https://arxiv.org/abs/2402.03885), [[Code]](https://github.com/moment-timeseries-foundation-model/moment)
- [x] **TTMs** - Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series. [[Arxiv 2024]](https://arxiv.org/pdf/2401.03955), [[Code]](https://huggingface.co/ibm-research/ttm-research-r2)
- [x] **GPT4TS** - One Fits All: Power General Time Series Analysis by Pretrained LM. [[NeurIPS 2023]](https://arxiv.org/abs/2302.11939), [[Code]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
- [x] **Time-LLM**: . Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. [[ICLR 2024]](https://arxiv.org/abs/2310.01728), [[Code]](https://github.com/KimMeen/Time-LLM)
- [x] **AutoTimes**: Autoregressive Time Series Forecasters via Large Language Models. [[NeurIPS 2024]](https://arxiv.org/abs/2402.02370), [[Code]](https://github.com/thuml/AutoTimes)
- [ ] LLMTime: Large Language Models Are Zero-Shot Time Series Forecasters. [[NeurIPS 2023]](https://arxiv.org/abs/2310.07820), [[Code]](https://github.com/ngruver/llmtime)
- [ ] Chronos: Learning the Language of Time Series. [[TMLR 2024]](https://arxiv.org/abs/2403.07815), [[Code]](https://github.com/amazon-science/chronos-forecasting)
- [ ] Time-MoE: Billion-Scale Time Series Foundation Models With Mixture Of Experts. [[ICLR 2025]](https://arxiv.org/abs/2409.16040), [[Code]](https://github.com/Time-MoE/Time-MoE)
- [ ] A Decoder-Only Foundation Model for Time-Series Forecasting. [[ICML 2024]](https://arxiv.org/abs/2310.10688), [[Code]](https://github.com/google-research/timesfm)


### Usage

1. Install Python 3.11 For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Place downloaded data in the folder ```./dataset```. Here is a [dataset summary](./figures/datasets.png).

- For univariate pre-training:
  * [UTSD](https://huggingface.co/datasets/thuml/UTSD) contains 1 billiion time points for large-scale pre-training (in numpy format): [[Download]](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
).
  * [ERA5-Familiy](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) (40-year span, thousands of variables) for domain-specific model: [[Download]](https://cloud.tsinghua.edu.cn/f/7fe0b95032c64d39bc4a/).

- For superwised training or modeling adaptation
  * Datasets from [TSLib](https://github.com/thuml/Time-Series-Library) : [[Download]](https://cloud.tsinghua.edu.cn/f/4d83223ad71047e28aec/).

1. We provide some supervised training, pre-training and adaptation scripts under the folder `./scripts/`:

```
# Supervised training
# (a) one-for-one forecasting
bash ./scripts/supervised/forecast/moirai_ecl.sh
# (b) one-for-all (rolling) forecasting
bash ./scripts/supervised/rolling_forecast/timer_xl_ecl.sh

# Large-scale pre-training
# (a) pre-training on UTSD
bash ./scripts/pretrain/timer_xl_utsd.sh
# (b) pre-training on ERA5
bash ./scripts/pretrain/timer_xl_era5.sh

# Model adaptation
# (a) full-shot fine-tune
bash ./scripts/adaptation/full_shot/timer_xl_etth1.sh
# (b) few-shot fine-tune
bash ./scripts/adaptation/few_shot/timer_xl_etth1.sh
```

4. Develop your large time-series model.

- Add the model file to the folder `./models`. You can follow the `./models/timer_xl.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

To evaluate zero-shot forecasting of large time-series models. Here we list some resources:
- Chronos: https://huggingface.co/amazon/chronos-t5-base
- Moirai: https://huggingface.co/Salesforce/moirai-1.0-R-base
- TimesFM: https://huggingface.co/google/timesfm-1.0-200m
- Timer-XL: https://huggingface.co/thuml/timer-base-84m
- Time-MoE: https://huggingface.co/Maple728/TimeMoE-50M
- TTMs: https://huggingface.co/ibm-research/ttm-research-r2

> [!NOTE]
> LTMs are still small in compared to foundation models of other modalities (for example, it is okay to use RTX 4090s for adaptation or A100s for pre-training).

### Citation

If you find this repo helpful, please cite our paper. 

```
@inproceedings{liutimer,
  title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
  author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  booktitle={Forty-first International Conference on Machine Learning}
}

@article{liu2024timer,
  title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
  author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2410.04803},
  year={2024}
}
```

### Acknowledgment

We appreciate the following GitHub repos a lot for their valuable code and efforts:
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Large-Time-Series-Model (https://github.com/thuml/Large-Time-Series-Model)
- AutoTimes (https://github.com/thuml/AutoTimes)

### Contributors

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)
* Haixuan Liu (liuhaixu21@mails.tsinghua.edu.cn)
