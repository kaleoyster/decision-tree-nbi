# Predictive modeling using decision tree

This repository contains code for the Decision Tree Model for the prediction the future condition and maintainance of the bridges. The implementation of the based on the bridge data submitted annually to Federal Highway Agency (FHWA) by the States, Federal agencies, and Tribal governments.

## Contents

1. Organization

   1. Configuration files

2. Data set
   1. Data source
   2. Format
3. Data acqusition
   1. Manual
   2. Automation
4. Models
   1. CNN
   2. VGG
5. Operations and use cases
6. Related repositories
   1. nbi-csv-json-converter
   2. nbi-datacenterhub
   3. nbe
   4. decision-tree-nbi

## Core documentation types

The following documentation is categorized into three documentation types:

**Concept**:
The concept documentation describes how and why the data processing or deep learning models works.

**Task**:
The task documentation provides specific instruction to execute to run specific models, or provide instructions to 
Gives specific instructions about how to get something done.
Tasks answer the question "how do I do it?".
When readers read tasks, they are doing something.
Tasks tend to have a specific goal and consist of a set of numbered steps that the reader can follow to achieve that goal.

**Reference**:
Contains structured information or specifications that users need to make a product work.
Reference material answers the question "what else do I need to know?"
When readers read references, they are fact-checking.
Reference sections should comprehensively catalog data such as functions and their parameters, return codes and error messages.
They are often presented as tables, bulleted lists, or sample scripts.

Our templates follow these documentation types, and you should find that your information naturally fits into them as you write.

## How to use these templates

We like to compare documentation types to aisles in a grocery store.
Each aisle includes related templates, which you can think of as ingredients.
Use these ingredients in documentation cookbooks to whip up docs for your readers.

When writing your documentation, it helps to think about:

* Who are you writing for?
* What will they be trying to do when they read the documentation?
* What information are you providing? Is it a concept, a task, or reference?

## The templates

Current templates:

| Template name | Documentation type | Description |
| ------------- | ------------------ | ----------- |
| [API Project overview](about-overview.md) | Concept | An overview of your API |
| [API Quickstart](api-quickstart/about-quickstart.md) | Concept, Task | Simplest possible method of implementing your API |
| [API Reference](https://github.com/thegooddocsproject/templates/blob/dev/api-reference/about-api-reference.md) | Reference | List of references related to your API |
| [Explanation](https://github.com/thegooddocsproject/templates/blob/dev/explanation/about-explanation.md) | Concept | Longer document giving background or context to a topic |
| [How-to](https://github.com/thegooddocsproject/templates/blob/dev/how-to/about-how-to.md) | Task | A concise set of numbered steps to do one task with the product. |
| [Tutorial](https://github.com/thegooddocsproject/templates/blob/dev/tutorial/about-tutorial.md) | Concept, Task | Instructions for setting up an example project using the product, for the purpose of learning. |
| [General reference entry](https://github.com/thegooddocsproject/templates/blob/dev/reference/about-reference.md) | Reference | Specific details about a particular topic |
| [Logging reference](https://github.com/thegooddocsproject/templates/blob/dev/logging/about-logging.md) | Reference | Description of log pipelines |




``` dot
digraph G {

  subgraph cluster_0 {
    style=filled;
    color=lightgrey;
    node [style=filled,color=white];
    a0 -> a1 -> a2 -> a3;
    label = "process #1";
  }

  subgraph cluster_1 {
    node [style=filled];
    b0 -> b1 -> b2 -> b3;
    label = "process #2";
    color=blue
  }
  start -> a0;
  start -> b0;
  a1 -> b3;
  b2 -> a3;
  a3 -> a0;
  a3 -> end;
  b3 -> end;

  start [shape=Mdiamond];
  end [shape=Msquare];
}
```

``` chart
{
  "type": "pie",
  "data": {
    "labels": [
      "Red",
      "Blue",
      "Yellow"
    ],
    "datasets": [
      {
        "data": [
          300,
          50,
          100
        ],
        "backgroundColor": [
          "#FF6384",
          "#36A2EB",
          "#FFCE56"
        ],
        "hoverBackgroundColor": [
          "#FF6384",
          "#36A2EB",
          "#FFCE56"
        ]
      }
    ]
  },
  "options": {}
}
```
## The cookbook ()

| Recipe name | Description |Constituent templates |
| ------- | ------- | ----------------- |
| API reference | One chapter in your full API documentation | Reference entries (multiple reference) + error information (reference) + throttling (concept) + authentication (task) |
| API guide: good | The starter set for API docs | API project overview + setup instructions (task) + Reference section (see recipe above) + Quickstart |
| API guide: better | Improved API docs, after learning about users | API project overview + setup instructions (task) + Reference(s) + Quickstart + How-to(s) |
