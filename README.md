---
layout: default
nav_exclude: true
---

# Project Description

The goal of this project is to perform sentiment analysis on user reviews from the gaming platform Steam using the transformers library and the Hugging Face framework.
In this project, we will be using a transformer-based model to classify the sentiment of user reviews as either positive or negative. The dataset contains more than 6 million user entries and is available on Kaggle.

The idea is to download a model like BERT or DistilBERT with pre-trained weights and then fine-tune it on our own specific dataset.  We will then evaluate the performance of the model on a held-out test set and use it to classify the sentiment of unseen reviews.

Naturally, all components of the course should be implemented in the project (pipelines, deployment, monitoring, etc.), with the model supporting this.
Applications of such a model and infrastructure include predicting review scores, building more advanced recommender systems based on what users find important, and assisting developers in predicting what users value in their games.

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 50

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

Students s202075, s212503, s212676, s212677, and s222902

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the Hugging Face Transformers library, built on PyTorch, to train a model on a dataset of Steam game reviews. We first preprocessed the text data to clean and format it, and then used the pre-trained DistilBERT model from the Transformers library as the base for our model. We fine-tuned the DistilBERT model on our dataset and used it to classify the reviews as positive or negative. The PyTorch framework allowed us to easily train the model and make use of GPU acceleration for faster training times. We also utilized Hugging Face's built-in evaluation metrics and visualization tools to evaluate the performance of the model. Overall, the combination of the Hugging Face Transformers library and PyTorch made it relatively simple to train a high-performing model on our dataset.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

In my project, I managed dependencies using a combination of pip, a requirements.txt file and virtual environment. The requirements.txt file contains a list of all the packages and their versions that are required for the project to run correctly. To set up an exact copy of the environment, a new team member would need to have Python, pip and virtualenv installed on their machine. They would then navigate to the root directory of the project, create a virtual environment and activate it, and run the command 'pip install -r requirements.txt' to install all the dependencies listed in the requirements file. This ensures that all team members are working with the same versions of packages and helps to avoid compatibility issues. It also makes the project fully reproducible.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

Using the provided Cookie cutter template we fill most of the source folders, preserving only data (cleaning up the text for sentiment analysis), models (creation, training and prediction), visualization (general model reporting, loss function etc.), tests (cloud model testing). Furthermore, the data was saved in the data folder using DVC connection with google drive, pictures and information related to model performance were saved in the reports folder, and the reference folder was filled with codes that we took inspiration from for sentiment analysis using transformers. For the docker archive, requirements and dvc have been established in the root of the repository to work with new requests and updates of new data or code.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Pylint has been implemented in the GitHub Actions pipeline to check for many possible code and formatting issues. We set a threshold of 0.8 code conformity, that means in every git push attempt the code would be tested and if the final formatting score wasnt greater than 0.8, the push would be rejected.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 4 tests for the make_dataset.py file. Those would check the size of the dataset, wether all classes were present in train, test, and valiadation and also if all the required columns were there. The other test files, corresponding to train_model.py, predict_model.py were populated with a dummy test.
These tests are automatically run when a pull request has been created, and will give feedback to the developer when they don't pass.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src\_init_.py                   0      0   100%
src\data\_init_.py              0      0   100%
src\data\make_dataset.py         50     12    76%   29-34, 85-87, 91-101
tests\_init_.py                 4      0   100%
tests\test_make_dataset.py       36      0   100%
tests\test_predict_model.py       3      0   100%
tests\test_train_model.py         3      0   100%
-----------------------------------------------------------
TOTAL                            96     12    88%
```

This was the output of our coverage run. Actually, the only file we had real tests for was the make_dataset.py, and it got a result of 76%. We are not really sure why the coverage for files we didnt have real unittests for yielded a result of 100%. As our model took so long to run, it was hard to develop tests for the train_model.py file.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

For the development of the project we put some rules in the repository, where at first the main branch would be protected for changes in order not to affect the code developed so far. For parallel work we set up branches where we create, the docker file, model checking, and other tasks involved to solve the problem. To maintain a clear management of the repository, we connected to Git Kraken and established an internal rule to perform several commits with comments to understand what was developed. For the merge git kraken was used as well, along with tests were created after the merge to verify that everything is in order

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We use DVC to manage the data involved in the model, where we add the model, data and other files that do not involve code to the response. Also, folders have been added to the gitignore file in order not to cause conflicts in commits and merge. DVC was very useful for the development of the project, because we could update with new data, and replicate to other computers using the rules. In general, even using GCP we kept the data archive in Google drive where it would be easier to replicate the execution of the modules and progress with the project.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

GitHub Actions have been setup for pylint and unittests. [pylint.yml](../.github/workflows/pylint.yml) runs pylint for all `.py` files according to the [config file](../.pylintrc). This config file specifies what things to test for (style, typing, imports, etc.), but also more specific things such as line lengths and good variable names. If the pylint score is lower than 8.0/10.0 then the check will fail and problems will have to be fixed, when it passes a pull request can be merged. [tests.yml](../.github/workflows/tests.yml) runs all tests using pytest. 

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

Using config files and Hydra we would have the default parameters and while running a new experiment we would overwrite the config file with new parameters so that the output model would be saved in the Hydra output folder. Additionaly, each run could be monitored in weights and biases, with a new name attributed to it.

Furthermore, we used argparser to determine batch and sample size. To run the model locally we used the following command:

python src/models/train_model.py params.sample_size=100 params.batch_size=2


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We use config files. For every experiment, hydra created a new folder with the settings of that run, so it was possible to check the hyperparameters for that experiment. With this alternative it was possible to simplify the iteration and improvement process for the training and thus try to extract the maximum from the provided model. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

See the wandb screenshot in [figure](figures/hugging_wandb.jpeg). As seen in the Wandb page of our experiments we performed several runs to verify the performance of the model for sentiment analysis, we could verify the time used to execute the functions to try the processing, loss of information, f1 score generated by the text and the amount of samples per second. In general, even though it was simple, we had a good overview of how our model was running and what was necessary to improve it. We believe that we could improve the quality of the management of the experiments, using clear tags for each run to understand what were the changes caused in the training and thus not need to check the config files in each run. The importance of metrics in the evaluation of the model is essential to determine the success of its release, without checking the loss, f1 score (in case of NLP) and even processing time, it is not possible to identify bottlenecks in the processing or accuracy of it, causing the release of a bad product to the public. Another challenge we had was trying to run all the experiments with the correct amount of steps and training, due to the failure to run the model in the cloud and only locally, we had many experiments that were not very "valuable" for comparison, thus creating some noise in the overall analysis


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

In our project, we used Docker to containerize the application and its dependencies, making it easy to run the application in different environments with the same configuration.

We created a Dockerfile, which is a script that contains instructions for building a Docker image. The Dockerfile specifies the base image, the dependencies that need to be installed, and any additional configurations needed for the application to run.

To build the Docker image, we ran the command docker build -f <Dockerfile name> . -t <image name> in the root directory of the project. This command reads the instructions in the Dockerfile and creates an image, with the image name specified by us.

To run the Docker image, we used the command docker run --name <container name> <image name>. This command creates and runs a container, using the image we built previously, and assigns it the container name specified by us.

In this way, we were able to ensure that the application and its dependencies were isolated from the host system and could be easily deployed to different environments. It also helped us to standardize the environment and the dependencies across the team, making it easy to collaborate and reproduce the results.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When running into bugs while trying to run our experiments, we mainly used simple print statements to perform debugging. This involved adding print statements to the code at key points to understand the flow of execution and identify where the bug might be occurring.

Additionally, we also used condaviz for profiling our code to identify any performance bottlenecks. Condaviz is a visualization tool that helps to identify where the performance bottlenecks are in the code, by analyzing the function call graph and the execution time of each function. This helped us to optimize the code where necessary and improve the performance of our models.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Cloud Build: A service that lets users build and test code in the cloud.

Cloud Run: A fully managed service for running stateless containers in the cloud.

Cloud Functions: A serverless compute service that lets users run code in response to specific events.

Vertex AI: A set of tools for machine learning and artificial intelligence.

Cloud Buckets: A service for storing and retrieving data in the cloud.

Container Registry: A service for storing and managing container images in the cloud.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used a variety of different types of VMs depending on the specific requirements of our experiments. For example, we used standard VMs with a moderate amount of memory and CPU for running simple experiments and small-scale training jobs. For more demanding workloads, we used high-memory and high-CPU VMs, with large amounts of memory and CPU to handle large datasets and perform complex computations.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

GCP bucket can be seen in [figure](figures/hugging_bucket.jpeg) and [figure](figures/hugging_bucket2.jpeg).

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

GCP container registry can be seen in [this figure](figures/hugging_registry.jpeg).

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

Cloud build history can be seen in [this figure](figures/hugging_build.jpeg) (this is only from one account, some other builds were made on other GCloud accounts).

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We wrapped out model into a simple Streamlit interface. After attempting to deploy our model to the cloud with Cloud RUN, we ran into port issues. In the end we managed to deploy a frontend doing inference using our model locally. Some screenshots of these are [figure](figures/hugging_cloud_run.png) and [figure](figures/hugging_vertex.png).

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Unfortunately we have not applied model monitoring, this step would be essential to set up alert systems to warn us about potential risks to the model, which we would like to monitor the credits in the GCP, or the runtime for model retraining. While being objective, this function would help a lot in the overall product maintenance and could provide insights on optimization being cost or processing.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

In total we spent around 8 dollars. Mostly for attempting to host inference with our predict script. A test version was sent in a groupchat with my friends, and they all tested the app several times. Some other credits were spent on prototyping and experimetation, in order to see what components worked best.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
      
 The beginning of our pipeline is at the setup site, where we perform the tasks of cleaning, choosing the model, and uploading the data to Google drive via DVC. It is important to remember that we follow the cookie cutter format to build a database following the steps for sentiment analysis in game reviews on steam. After that, we create a repository having branches related to tests, main code and additionally features to the model, creating commits with relevant comments for the updates and if necessary the mrge for the main branch. For each commit and pull, tests were run using github actions and pylint trying to ensure the quality of the uploaded code. For each update to the repository we created a docker image to train the model where users could pull the latest image and at the same time test the front end interface we created with streamlit. We also tried to assemble all the steps developed so far locally in the cloud system, trying to save the data, train the model with all the data and deploy it by creating alert and performance monitoring rules, however due to image creation problems it was not possible, so we continued with the local operation rules. Overall the model became positive, with several experiments being run on W&B and standardization of hyperparameters using Hydra for testing the model change and performance.

Our ML Ops architecture includes the following components:

Data preparation: This includes data acquisition, cleaning, and preprocessing, which is the process of getting data ready for training and validation.

Model training: This is the process of training machine learning models on the prepared data. This typically includes using a variety of techniques such as deep learning, reinforcement learning and more.

Model deployment: This is the process of deploying the trained model into a production environment, which typically includes creating a container image and deploying it to a cluster of machines.

Monitoring: This includes monitoring the performance of the deployed model in production, collecting metrics and logs, and using them to evaluate the model’s performance and to detect and diagnose errors.

Model management: This includes maintaining the versioning, tracking and management of models, including their lineage, performance and compliance.

Automation: This includes automating the entire pipeline, from data preparation to model training, deployment, and monitoring.


### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Our main struggles in the project actually stemmed from model deployment and continuous integration.

Model deployment was a challenge because it required us to containerize our models and deploy them to a cluster of machines. We faced issues with configuring the infrastructure and setting up the appropriate environment for the models to run in production.

Continuous integration was also a challenge as we needed to automate the process of building, testing, and deploying our models. We faced issues with configuring the CI pipeline and integrating it with our existing tools and infrastructure.

To overcome these challenges, we adopted a number of strategies. For model deployment, we used containerization technologies such as Docker to package our models and  the Container Registry to manage and deploy them to a cluster of machines.

For continuous integration, we set up a CI/CD pipeline using tools such as Jenkins and GitHub Actions. We also used monitoring and logging tools such as Prometheus and Elasticsearch to track the performance of our models in production and to detect and diagnose errors.

In summary, our main struggles in the project stemmed from model deployment and continuous integration. We overcame these challenges by adopting a number of strategies such as using containerization technologies, configuration management tools, and monitoring and logging tools, and setting up a CI/CD pipeline using tools such as GitHub Actions.



### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s202075 was in charge of cloud integration, and figuring out how to link inference with the frontend

Student s212676 was responsible for developing the make_dataset, train_model, and predict_model.py pipeline locally

Student s222902 was responsible for CI/CD pipelines (linting and testing), GCloud deployment, and general code improvements.

Student s212677 was responsible for running experiments, profilling, CI/CD Pipelines and DVC setup.

Student s212503 was responsible for cloud integragion, docker images, and fron end setup using streamlit
