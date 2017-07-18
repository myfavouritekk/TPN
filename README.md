# TPN: Tubelet Proposal Network

## Introduction
`TPN`, short for `Tubelet Proposal Network`, is a deep learning framework for video object detection, originally designed for ImageNet VID dataset.

This framework mainly contains two components shown below:
![TPN Framework](tpn.png)

The `Tubelet Proposal Network` generates hundreds of tubelet proposals based on static image box proposals, and the `Encoder-decoder LSTM` encodes the tubelet visual features into memory and decodes the information to classify each tubelet box proposal into different classes. More details in the [paper](https://arxiv.org/pdf/1702.06355) in `CVPR 2017`.

## Citation
If you are using the `TPN` code in your project, please cite the following [publication](https://arxiv.org/pdf/1702.06355).

```latex
@inproceedings{kang2017tpn,
  title={Object Detection in Videos with Tubelet Proposal Networks},
  author={Kang, Kai and Li, Hongsheng and Xiao, Tong and Ouyang, Wanli and Yan, Junjie and Liu, Xihui and Wang, Xiaogang},
  Booktitle = {CVPR},
  Year = {2017}
}
```

## Installations

### Dependencies
1. [Faster R-CNN]()
2. [vdetlib]()
3. [Caffe]() with MPI
4. [TensorFlow]() v0.8
### Instructions

```sh
>> # clone this repository
>> git clone --recursive git@github.com:myfavouritekk/TPN.git
>> # compile external dependencies
>> cd $TPNROOT/external/vdetlib && make
>> cd $TPNROOT/external/py-faster-rcnn/lib/ && make
>> # compile caffe with MPI
>> cd $TPNROOT/external/caffe-mpi
>> mkdir build && cd build
>> cmake .. && make -j
```

## Demo

```sh
>> bash demo.sh
>> 
```

## Beyond demo
1. Generating the `.vid` file for a video
```sh
>> python external/vdetlib/tools/gen_vid_proto_file.py vid_name root_dir out_file.vid
```
2. Generating the `.box` file for region proposals
proposal files contains two fields: `boxes` contains box proposals, `images` contains image frame names. The `images` can be the following forms: `video_name/frame_name`, `subset/video_name/frame_name`.
```sh
>> python tools/data/box_proto_from_proposals.py proposal_file vid_root save_dir
```

## Models (coming soon...)
1. Static region proposal models
2. TPN models
3. ED-LSTM models

## Links
1. [PDF](https://arxiv.org/pdf/1702.06355)
2. Poster
3. Demo video

## License
`TPN` is released under the MIT License.


