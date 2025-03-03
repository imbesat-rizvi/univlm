# Ilia Davydov Weekly Reports

## Student Information
- **Name**: Ilia Davydov
- **Project Title**: UniVLM -- A unified Vision-Language Model (VLM) library
- **Mentor**: Md Imbesat Hassan Rizvi
- **Course**: Data Analysis Software Project (DASP) for Natural Language Processing (NLP) 2024-25 -- 9 ECTS

---

## Reports

### Week [KW 46]

- **Update 1**: We have started working on our project, held several initial meetings, and assigned tasks
- **Update 2**: I have read the recommended research papers for the project (there were three of them) and tried to understand them as much as possible
- **Update 3**: We divided the tasks, and I was assigned the Task Unification task
- **Next Steps**: to think about how we will implement our task and its structure.

### Week [KW 47]

- **Update 1**: We divided the work on our task as follows: each team member is assigned a specific model, and I took on the task of integrating SAM-v2 into our project.
- **Update 2**: At home, I thought about how we could implement our project and presented my vision during the weekly meeting
![image](https://github.com/user-attachments/assets/313cd19c-04f5-475c-b69d-e9644f9b38bb)

- **Update 3**: We discussed my vision and made adjustments after reaching a consensus. We decided to split the task into two stages: first, we will create a pipeline as a Python module for professional developers so they can work with it as developers. Once that is implemented, we will integrate it into a user-friendly interface for end-users without programming knowledge.
- **Next Steps**: To start the implementation of the pipeline

### Week [KW 48]

- **Update 1**: I started studying the SAM-v2 model in more detail: reviewing the code, testing its functionality, analyzing the parameters, inputs, and outputs of the functions, exploring and applying the usage instructions, as well as working with demonstration examples in Google Colab
- **Update 2**: I started implementing the pipeline, focusing on my assigned model. Later, we plan to integrate the other models that my teammates are working on in parallel
- **Next Steps**: to finish our pipline 

### Week [KW 49]

- **Update 1**: I wrote the first version of the handler for our model, which provided a user-friendly interface for easy interaction with our model. It allowed users to simply select a point on the image, click on it, and get the 3 best masks with scores. To achieve this, I extracted the handler functions from the original model into a separate file and also wrote my own functions
- **Update 2**: I started to prepare to our mid-term presentation
- **Next Steps**: Fix my model so that it includes more advanced features, such as excluding a certain area from segmentation or working with video (since SAM-2 is capable of this), and also work on our presentation and present it

### Week [KW 50]

- **Update 1**: Next week, we dedicated our time to preparing for the presentation and gave it.
- **Update 2**: I also prepared a video for my team to explain my part of the work and to present the concept of what I would be talking about during the presentation (it was just a video; in the end, we didn't include it in the presentation, and I presented it myself)
- **Update 3**: [video](./sam2-video_kzbX1yPF.mp4)

- **Next Steps**: Discuss our presentation at the last meeting of the year

### Week [KW 51]

- **Update 1**: Unfortunately, a force majeure situation occurred, and the person responsible for the Marigold model is no longer available for the project. Since the work on integrating SAM2 into the pipeline was still in its early stages and far from the final version, as well as being quite complex (due to the model’s multifunctionality, such as working with video, among other things), we decided that in order to meet the project deadline and given that our team has been reduced, I will take over the work on Marigold. If we have enough time, we will return to SAM2 later.
- **Update 2**: Also, at the last meeting following the presentation, we decided that we lacked a clear structure. Despite the fact that everyone studied their model well, each of us integrated them into the pipeline in different ways, and we should have come to something more standardized

- **Next Steps**: Study Marigold on Hugging Face

- ### Week [KW 52]

- **Update 1**: Christmas & New Year break at TU Darmstadt.

- - ### Week [KW 1]

- **Update 1**: Christmas & New Year break at TU Darmstadt.

- - ### Week [KW 2]

- **Update 1**: Christmas & New Year break at TU Darmstadt.

- ### Week [KW 3]

- **Update 1**: At the first meeting in 2025, we discussed the features of our models and the general requirements for integrating them into our pipeline. We proposed our structures and code examples, and one common pipeline structure was chosen, to which all models should be aligned

- **Next Steps**: Integrate my model (Marigold) into the structure of our pipeline

### Week [KW 4]

- **Update 1**: I have thoroughly studied the entire proposed structure and integrated the solution for Marigold into our pipeline
- **Update 2**: I pushed these changes to GitHub, and thus we now have a new version of the project
- **Update 3**: During the meeting, we discussed the issues of our pipeline with Rizvi and set a goal to develop a new, improved structure

- **Next Steps**: To enhance our pipeline and make it even more universal
- **Next Steps**: Start preparing the poster for the presentation

### Week [KW 5]

- **Update 1**: Discussions about our structure, joint additional meetings with Aryan for its explanation and accelerated understanding
- **Update 2**: Unfortunately, a second team member has left our project, and now only three of us remain. Because of this, we decided to focus on HF models for now, while treating models like Apple DepthPro and Marigold as exceptions by writing custom classes for them ourselves
- **Update 3**: Kick-off the poster preparation for the presentation

- **Next Steps**: Implement classes for Marigold in the final pipeline
- **Next Steps**: Finalize the poster preparation for the presentation

  ### Week [KW 6]

- **Update 1**: All exceptional classes were written, and our first complete version of the pipeline was presented to our mentor, Rizvi. It was not final, but already sufficient for us to start preparing it for submission at the end of the project and improving it
- **Update 2**: Additional meetings and discussions about our presentation
- **Update 3**: Creating the first version of the documentation and adding it to our poster
- **Update 4**: Finalization, printing, and presentation of the poster
  
- **Next Steps**: Discuss the next steps with Rizvi.
This week was too intense due to the presentation, so we focused on it rather than on the code and project structure. Because of this, code improvements have been on pause for now

  ### Week [KW 7]

- **Update 1**: We discussed and focused on finalizing our project: testing and code optimization
- **Update 2**: Separately, I started improving the documentation and preparing it for the final submission at the end of February.
  
- **Next Steps**: Discuss the report writing

  ### Week [KW 8]

- **Update 1**: Continuation of project finalization: testing and code optimization
- **Update 2**: Continuation of documentation improvement
- **Update 3**: Start of report creation by our team
  
- **Next Steps**:Take the final steps in all areas of our project for submission and ensure quality control

  ### Week [KW 9]

- **Update 1**: Final touches in documentation, optimization, testing, and the report.
- **Update 2**: Project submission



