# ai-model-evaluation-KF5042
Repository for model evaluation for KF5042
Author: Viktorija Ivanova

This has to be entered into Terminal to install necessary libraries
```
pip install transformers datasets torch torchvision pycocotools
```

Then run evaluation.py

Firstly 10 images with detected objects will appear on the screen with boxes and labels, this is for previewing the output. 
After you inspect and close them the next part of the code will run, going through 5000 images from dataset ("val" subset only) and evaluating the model, outputting the precision-recall curve graph.
