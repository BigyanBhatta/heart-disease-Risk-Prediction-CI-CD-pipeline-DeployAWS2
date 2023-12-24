
# Heart Stroke/Disease Risk Prediction

### Purpose

The term “heart disease” refers to several types of heart conditions. The most common type of heart disease in the United States is coronary artery disease (CAD), which affects the blood flow to the heart. Decreased blood flow can cause a heart attack.

High blood pressure, high blood cholesterol, and smoking are key risk factors for heart disease. About half of people in the United States (47%) have at least one of these three risk factors.2 Several other medical conditions and lifestyle choices can also put people at a higher risk for heart disease, including overweight (obesity), unhealthy diet, physical inactivity, excessive alcohol use, etc.

The project aim is to develop a prediction model for Heart stroke/disease chances based on the health and lifestyle parameters. The parameters taken for the study and their categories and range is shown below: 

- gender = 0 (Male), 1 (Female)

- age = 32..70

- education = [(0)'uneducated'], [(1)'primaryschool'], [(2)'graduate'], [(3)'postgraduate']

- currentSmoker = 0 (No), 1 (Yes)

- cigs_per_day = Number of cigarettes per day = 0...70

- bp_meds = Whether or not the person is taking blood pressure drugs = 0 (No), 1 (Yes)

- prevalent_stroke = Whether or not the person has had a stroke = 0 (No), 1 (Yes)

- prevalentHyp = Blood pressure (BP) > = 140 mm Hg systolic and/or > = 90 diastolic = 0 (No), 1 (Yes)

- diabetes = Whether or not the person has diabetes = 0 (No), 1 (Yes)

- tot_chol = Total cholesterol = 113...464

- sysBP = Systolic blood pressure = 84...220

- diaBP = Diastolic blood pressure = 48...140

- BMI = 16...57

- heartRate = Heart rate = 44...143

- glucose = Blood Sugar = 40...394

- Heart_stroke = Whether or not the person has had a heart attack = 0 (No), 1 (Yes)

The dataset used for the project is based on Heart Disease dataset, containing medical records of around 4,200 patients.


### How to run?
STEPS:

1. Clone the repository
```http
 git clone <giturl>
```
2. Create a conda environment after opening the repository
```http
  conda create -n project-venv python=3.8 
```
3. Then activate the created environment
```http
  conda activate project-venv
```
4. Install all the libraries and packages required for running the project
```http
  pip install -r requirements.txt
```
5. To finally run on the local host
```http
  python app.py
```

### Github Setup

```http
git init 
```
```http
git add . 
```

```http
git commit -m "commit-name"
```
```http
git branch -M branch-name
```
```http
git remote origin <git_url>
```
```http
git push origin branch-name
```







## Demo

https://github.com/BigyanBhatta/project_ml/assets/143421101/098579b2-e427-4e89-a092-1f9d2c7fe8d7


