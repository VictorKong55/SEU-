仅记录人工智能大作业的项目完成经历。
问题描述：
<img width="531" alt="截屏2024-05-30 20 16 51" src="https://github.com/VictorKong55/SEU-/assets/171250754/2bb519b7-1323-40a4-903d-70186eab2f9f">
我的完成思路如下：
（1）观察数据类型；
（2）选用合适神经网络；
（3）进行训练、验证、测试。
（1）观察数据类型：
<img width="822" alt="截屏2024-05-30 20 19 44" src="https://github.com/VictorKong55/SEU-/assets/171250754/34a2f24a-e6be-4bcc-9c8b-6e06513808e6">
（2）选用合适神经网络：
<img width="768" alt="截屏2024-05-30 20 20 32" src="https://github.com/VictorKong55/SEU-/assets/171250754/e62134e8-1f8a-4945-ab00-25312040845c">
（3）进行训练、验证、测试，这里给出几个例子：
![train](https://github.com/VictorKong55/SEU-/assets/171250754/5e0e4ef4-ff06-4681-bb90-6fdc0395ab22)
![pred 1 1](https://github.com/VictorKong55/SEU-/assets/171250754/cd4c1b62-2a90-466b-9f22-85045317fb88)
![pred1 2](https://github.com/VictorKong55/SEU-/assets/171250754/4e0032c4-b94d-4c7b-ba5a-97e16520da7e)
![pred 2](https://github.com/VictorKong55/SEU-/assets/171250754/33994a8f-01d6-466a-ae26-f99abbb0576e)
![loss](https://github.com/VictorKong55/SEU-/assets/171250754/3460808f-f56f-4c13-95c4-9eb607bdbc99)
问题思考：有两组数据的拟合效果不是很好，这是为什么？
回答如下：结构固有频率为2.5HZ，所以当荷载频率也为2.5HZ时，会出现共振，导致拟合出现问题。
解决办法：提供2.5HZ左右的F-X数据，并将2.5HZ附近的数据进行单独训练。
