echo # CNN for PneumoniaMNIST > README.md
echo. >> README.md
echo Lightweight CNN for binary classification of chest X-rays ^(Normal vs. Pneumonia^) using MedMNIST. >> README.md
echo. >> README.md
echo ## Architecture >> README.md
echo Conv2D^(16^) -^> MaxPool -^> Conv2D^(32^) -^> MaxPool -^> Flatten -^> Dense^(1, sigmoid^) >> README.md
echo. >> README.md
echo ## Results >> README.md
echo - Accuracy: 0.8526 >> README.md
echo - Precision: 0.8184 >> README.md
echo - Recall: 0.9821 >> README.md
echo - F1-Score: 0.8928 >> README.md
echo. >> README.md
echo ## Setup >> README.md
echo pip install -r requirements.txt >> README.md
echo python data_prep.py >> README.md
echo python train.py >> README.md
echo python evaluate.py >> README.md
