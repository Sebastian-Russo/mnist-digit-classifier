## LR = 0.0001

Using device: cpu
Loading MNIST dataset...

Model architecture:
DigitClassifier(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (relu): ReLU()
)

Starting training...
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.306080
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.876382
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.596106
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.373709
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.341845
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.233442
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.468613
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.462502
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.238997
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.163679

Test set: Average loss: 0.1386, Accuracy: 9586/10000 (95.86%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.347572
Train Epoch: 2 [6400/60000 (11%)]       Loss: 0.283496
Train Epoch: 2 [12800/60000 (21%)]      Loss: 0.171347
Train Epoch: 2 [19200/60000 (32%)]      Loss: 0.171648
Train Epoch: 2 [25600/60000 (43%)]      Loss: 0.178529
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.175324
Train Epoch: 2 [38400/60000 (64%)]      Loss: 0.250232
Train Epoch: 2 [44800/60000 (75%)]      Loss: 0.346606
Train Epoch: 2 [51200/60000 (85%)]      Loss: 0.097526
Train Epoch: 2 [57600/60000 (96%)]      Loss: 0.045279

Test set: Average loss: 0.0848, Accuracy: 9739/10000 (97.39%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.120293
Train Epoch: 3 [6400/60000 (11%)]       Loss: 0.115761
Train Epoch: 3 [12800/60000 (21%)]      Loss: 0.090345
Train Epoch: 3 [19200/60000 (32%)]      Loss: 0.130159
Train Epoch: 3 [25600/60000 (43%)]      Loss: 0.049448
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.177776
Train Epoch: 3 [38400/60000 (64%)]      Loss: 0.180604
Train Epoch: 3 [44800/60000 (75%)]      Loss: 0.080996
Train Epoch: 3 [51200/60000 (85%)]      Loss: 0.088541
Train Epoch: 3 [57600/60000 (96%)]      Loss: 0.245428

Test set: Average loss: 0.0638, Accuracy: 9802/10000 (98.02%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.094373
Train Epoch: 4 [6400/60000 (11%)]       Loss: 0.051172
Train Epoch: 4 [12800/60000 (21%)]      Loss: 0.103525
Train Epoch: 4 [19200/60000 (32%)]      Loss: 0.166601
Train Epoch: 4 [25600/60000 (43%)]      Loss: 0.123328
Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.100939
Train Epoch: 4 [38400/60000 (64%)]      Loss: 0.039496
Train Epoch: 4 [44800/60000 (75%)]      Loss: 0.167488
Train Epoch: 4 [51200/60000 (85%)]      Loss: 0.055855
Train Epoch: 4 [57600/60000 (96%)]      Loss: 0.091639

Test set: Average loss: 0.0500, Accuracy: 9838/10000 (98.38%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.027906
Train Epoch: 5 [6400/60000 (11%)]       Loss: 0.325094
Train Epoch: 5 [12800/60000 (21%)]      Loss: 0.070178
Train Epoch: 5 [19200/60000 (32%)]      Loss: 0.011845
Train Epoch: 5 [25600/60000 (43%)]      Loss: 0.028163
Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.131563
Train Epoch: 5 [38400/60000 (64%)]      Loss: 0.092592
Train Epoch: 5 [44800/60000 (75%)]      Loss: 0.158590
Train Epoch: 5 [51200/60000 (85%)]      Loss: 0.147183
Train Epoch: 5 [57600/60000 (96%)]      Loss: 0.042211

Test set: Average loss: 0.0434, Accuracy: 9857/10000 (98.57%)

Model saved as mnist_model.pth
Saved predictions visualization to predictions.png
Saved training history to training_history.png

Final Test Accuracy: 98.57%

### Conclusion
- What happened: The model is taking tiny baby steps!
- Analogy: Detectives are being overly cautious:
- "We got it wrong? Let's make a microscopic adjustment..."
- "Hmm, still not perfect. Another tiny adjustment..."
- Progress is happening, but very slowly
- Steady improvement each epoch (good!)
- BUT improvement is small (95.86% → 98.57% over 5 epochs)
- With more epochs (10-20), it would probably reach 99%+
- Just needs more time!


## LR = 0.001

Using device: cpu
Loading MNIST dataset...

Model architecture:
DigitClassifier(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (relu): ReLU()
)

Starting training...
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.329343
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.279436
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.210591
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.190610
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.116041
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.181418
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.180926
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.096462
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.047877
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.052928

Test set: Average loss: 0.0472, Accuracy: 9841/10000 (98.41%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.213999
Train Epoch: 2 [6400/60000 (11%)]       Loss: 0.171410
Train Epoch: 2 [12800/60000 (21%)]      Loss: 0.043678
Train Epoch: 2 [19200/60000 (32%)]      Loss: 0.104787
Train Epoch: 2 [25600/60000 (43%)]      Loss: 0.037711
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.057384
Train Epoch: 2 [38400/60000 (64%)]      Loss: 0.175552
Train Epoch: 2 [44800/60000 (75%)]      Loss: 0.062415
Train Epoch: 2 [51200/60000 (85%)]      Loss: 0.100295
Train Epoch: 2 [57600/60000 (96%)]      Loss: 0.096367

Test set: Average loss: 0.0299, Accuracy: 9898/10000 (98.98%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.093676
Train Epoch: 3 [6400/60000 (11%)]       Loss: 0.130621
Train Epoch: 3 [12800/60000 (21%)]      Loss: 0.264257
Train Epoch: 3 [19200/60000 (32%)]      Loss: 0.116496
Train Epoch: 3 [25600/60000 (43%)]      Loss: 0.021567
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.002505
Train Epoch: 3 [38400/60000 (64%)]      Loss: 0.243256
Train Epoch: 3 [44800/60000 (75%)]      Loss: 0.115310
Train Epoch: 3 [51200/60000 (85%)]      Loss: 0.025324
Train Epoch: 3 [57600/60000 (96%)]      Loss: 0.039398

Test set: Average loss: 0.0306, Accuracy: 9897/10000 (98.97%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.063724
Train Epoch: 4 [6400/60000 (11%)]       Loss: 0.104453
Train Epoch: 4 [12800/60000 (21%)]      Loss: 0.051929
Train Epoch: 4 [19200/60000 (32%)]      Loss: 0.030451
Train Epoch: 4 [25600/60000 (43%)]      Loss: 0.109258
Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.014656
Train Epoch: 4 [38400/60000 (64%)]      Loss: 0.031682
Train Epoch: 4 [44800/60000 (75%)]      Loss: 0.037930
Train Epoch: 4 [51200/60000 (85%)]      Loss: 0.004349
Train Epoch: 4 [57600/60000 (96%)]      Loss: 0.151845

Test set: Average loss: 0.0259, Accuracy: 9913/10000 (99.13%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.046386
Train Epoch: 5 [6400/60000 (11%)]       Loss: 0.101315
Train Epoch: 5 [12800/60000 (21%)]      Loss: 0.003742
Train Epoch: 5 [19200/60000 (32%)]      Loss: 0.135939
Train Epoch: 5 [25600/60000 (43%)]      Loss: 0.006527
Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.034997
Train Epoch: 5 [38400/60000 (64%)]      Loss: 0.015511
Train Epoch: 5 [44800/60000 (75%)]      Loss: 0.011225
Train Epoch: 5 [51200/60000 (85%)]      Loss: 0.020873
Train Epoch: 5 [57600/60000 (96%)]      Loss: 0.008801

Test set: Average loss: 0.0239, Accuracy: 9921/10000 (99.21%)

Model saved as mnist_model.pth
Saved predictions visualization to predictions.png
Saved training history to training_history.png

Final Test Accuracy: 99.21%

### Conclusion
- What happened: Perfect balance of speed and stability!
- Analogy: Detectives make smart, measured adjustments:
- "We made a mistake? Let's analyze and make a reasonable correction"
- Fast enough to learn quickly
- Careful enough not to overcorrect
- Fast initial learning (98.41% after just 1 epoch!)
- Steady, stable improvement
- Low test loss (0.0239) = confident, accurate predictions


## LR = 0.01

Using device: cpu
Loading MNIST dataset...

Model architecture:
DigitClassifier(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (relu): ReLU()
)

Starting training...
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.310461
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.278549
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.388047
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.142192
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.330730
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.271435
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.275761
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.095109
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.342127
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.219308

Test set: Average loss: 0.1108, Accuracy: 9651/10000 (96.51%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.173342
Train Epoch: 2 [6400/60000 (11%)]       Loss: 0.221066
Train Epoch: 2 [12800/60000 (21%)]      Loss: 0.102321
Train Epoch: 2 [19200/60000 (32%)]      Loss: 0.131991
Train Epoch: 2 [25600/60000 (43%)]      Loss: 0.353814
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.288367
Train Epoch: 2 [38400/60000 (64%)]      Loss: 0.036531
Train Epoch: 2 [44800/60000 (75%)]      Loss: 0.159321
Train Epoch: 2 [51200/60000 (85%)]      Loss: 0.336089
Train Epoch: 2 [57600/60000 (96%)]      Loss: 0.272210

Test set: Average loss: 0.1053, Accuracy: 9689/10000 (96.89%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.347037
Train Epoch: 3 [6400/60000 (11%)]       Loss: 0.171232
Train Epoch: 3 [12800/60000 (21%)]      Loss: 0.084505
Train Epoch: 3 [19200/60000 (32%)]      Loss: 0.408265
Train Epoch: 3 [25600/60000 (43%)]      Loss: 0.158349
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.160332
Train Epoch: 3 [38400/60000 (64%)]      Loss: 0.301891
Train Epoch: 3 [44800/60000 (75%)]      Loss: 0.061792
Train Epoch: 3 [51200/60000 (85%)]      Loss: 0.151634
Train Epoch: 3 [57600/60000 (96%)]      Loss: 0.224554

Test set: Average loss: 0.0953, Accuracy: 9728/10000 (97.28%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.163924
Train Epoch: 4 [6400/60000 (11%)]       Loss: 0.163599
Train Epoch: 4 [12800/60000 (21%)]      Loss: 0.177223
Train Epoch: 4 [19200/60000 (32%)]      Loss: 0.167957
Train Epoch: 4 [25600/60000 (43%)]      Loss: 0.173570
Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.168809
Train Epoch: 4 [38400/60000 (64%)]      Loss: 0.192857
Train Epoch: 4 [44800/60000 (75%)]      Loss: 0.099566
Train Epoch: 4 [51200/60000 (85%)]      Loss: 0.142586
Train Epoch: 4 [57600/60000 (96%)]      Loss: 0.141297

Test set: Average loss: 0.0944, Accuracy: 9737/10000 (97.37%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.118086
Train Epoch: 5 [6400/60000 (11%)]       Loss: 0.064123
Train Epoch: 5 [12800/60000 (21%)]      Loss: 0.268489
Train Epoch: 5 [19200/60000 (32%)]      Loss: 0.169939
Train Epoch: 5 [25600/60000 (43%)]      Loss: 0.273670
Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.231723
Train Epoch: 5 [38400/60000 (64%)]      Loss: 0.179092
Train Epoch: 5 [44800/60000 (75%)]      Loss: 0.326304
Train Epoch: 5 [51200/60000 (85%)]      Loss: 0.198009
Train Epoch: 5 [57600/60000 (96%)]      Loss: 0.239953

Test set: Average loss: 0.1206, Accuracy: 9666/10000 (96.66%)

Model saved as mnist_model.pth
Saved predictions visualization to predictions.png
Saved training history to training_history.png

Final Test Accuracy: 96.66%

### Conclusions
- What happened: The model is taking giant steps during learning!
- Analogy: The detectives are overreacting to every mistake:
- Day 1: "We got a 7 wrong? COMPLETELY CHANGE EVERYTHING!"
- Day 2: "Wait, now we're worse! CHANGE EVERYTHING AGAIN!"
- Day 5: "We keep overcorrecting and getting worse!"
- High variance in losses (0.064 to 0.326 in same epoch!)
- Test loss going UP instead of down
- The model is thrashing - making progress then undoing it




## 🎯 Final Verdict

Metric          lr=0.0001           lr=0.001            ⭐lr=0.01
Speed🐌         Slow🚀              Fast⚡               Too fast
Stability       ✅ Stable           ✅ Stable           ❌ Unstable
Final Result😊  Good🎉              Excellent😕         Mediocre

Verdict "Would work with 10+ epochs""Perfect!"          "Needs reduction"
