# Siddhant Tyagi Weekly Reports

## Student Information
- **Name**: Siddhant Tyagi
- **Project Title**: UniVLM -- A unified Vision-Language Model (VLM) library
- **Mentor**: Md Imbesat Hassan Rizvi
- **Course**: Data Analysis Software Project (DASP) for Natural Language Processing (NLP) 2024-25 -- 9 ECTS

---

## Reports

### Week [2]

- **Update 1**: Implemented a working approach where I first cached 100000 models name using HFapi() and then the user query in processed parallely in 5 partitions of data and top 5 matched models are returned and option to choose one of them is given 
- **Update 2**: Thought for next approach would be to use model hugging face page as a document which would be used to feed into code generation which will be used then to generate the inference script for the model and then it will be used for processing the input given
- **Challenges**: Finding a good model for purpose and optimizing the generation properly
- **Next Steps**: Testing different model for the task

### Week [1]

- Researched similar workflows to our unified inference API to gather inspiration and motivation 
- Clarified Doubts to the mentor and then made a rough workflow regarding the approach.
- **Challenges**: How to have the models be loaded and how to make it in a scalable manner  
- **Next Steps**: Implementing the starting point for the project which is querying the model name and getting whether the model exists or not on huggingface and fuzzy matching the name to make it more user-convenient 

### Week [0]

- Understanding Problem Statement discussion regarding technicality and expectations 
- Distribution of the workflow among the team 

---
