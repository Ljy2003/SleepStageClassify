# 思路
把整个数据集分成很多段，每段30秒，在每30秒的时间段内做一个snn+transformer分类，输入transformer的序列为每一秒内各个频段的脉冲次数。
# 已经完成的
- 数据用https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/#files-panel
- 滤波分成5个频段+spike编码
- SNN处理spike去除伪影
- 可视化展示原始spike编码，以及经过SNN的spike
![Alt text](output/images/origional_spike/2.png)<center>原始spike</center>
![Alt text](output/images/snn_spike/2.png)<center>经过SNN后的spike</center>

# 问题
SNN运行很慢，仿真一个30秒需要3分钟多