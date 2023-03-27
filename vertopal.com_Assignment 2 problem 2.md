---
output:
  md_document: default
  pdf_document: default
title: Data Mining Assignment 2 problem 2
---

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

**Problem 2**

Dataset: german_credit.csv

\`\`\`{r echo=FALSE, warning=FALSE, message=FALSE} library(dplyr)
library(tidyverse)

german_credit \<-
read.csv("https://raw.githubusercontent.com/jgscott/ECO395M/master/data/german_credit.csv",
sep = ",", dec = ".")

# a bar plot of default probability by credit history

german_credit %\>% group_by(history) %\>% summarise(default_probability
= mean(Default)) %\>% ggplot(aes(x=history, y=default_probability)) +
geom_bar(stat="identity")


    Figure 1. A bar plot of default probability by credit history

    ```{r echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
    # a logistic regression model for predicting default probability
    model <- glm(Default ~ duration + amount + installment + age + history + purpose + foreign, 
             data = german_credit,family = binomial)
    coef(model)

We build a logistic regression model for predicting default probability,
using the variables duration + amount + installment + age + history +
purpose + foreign. The results shows that the probability of default is
higher for a person with good history in comparison to a person with
poor or terrible history. This result is counterintuitive, and we think
that this data set is not appropriate for building a predictive model of
defaults. That is because the sample is biased, and we could use
randomized sample to get correct model.
