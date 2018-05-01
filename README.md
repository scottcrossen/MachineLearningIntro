# Machine Learning Intro

A simple assortment of basic machine learning algorithms

### Description

This repository is maintained in Java and Scala and designed as a simple demonstration of machine learning algorithms.
It will loop through a set of algorithms for a given ARRF file and display that algorithm's results for subsequent trials.
Though a few classes are written in Java (such as the instructor-provided toolkit), the majority of the code is written
in Scala.

This project provides two docker-containers: a JVM container and a
Maven container. One compiles the code and one runs the code whenever it detects a new Jar file. Major services are
containerized using docker's \'docker-compose\' cluster management.

Many of the files contained in this repository also fullfill some of the requirements for the BYU course titled CS 478.
It is written and maintained by Scott Leland Crossen.

### Getting Started

#### Development Instructions

To start your machine learning program:
1. Make sure that the program 'docker-compose' is installed on your machine.
2. Clone this repository
3. Navigate to the 'docker' folder
4. Use the command ```docker-compose up``` to start the cluster
5. Additionally, you can attach to individual containers with the command ```docker-compose exec [container_name] bash```

### Contributors

1. Scott Leland Crossen  
<http://scottcrossen.com>  
<scottcrossen42@gmail.com>
