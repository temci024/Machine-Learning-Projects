.. Customer Spent Analysis using K-Means Clustering documentation master file, created by
   sphinx-quickstart on Wed Apr 17 20:19:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Customer Spent Analysis using K-Means Clustering
============================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Project Overview:
-------------------
The "Customer Spent Analysis using K-Means Clustering" project utilizes K-Means clustering to segment customers based on income and spending patterns. 
It employs Streamlit for the interface, pandas for data manipulation, and scikit-learn for model training. 
The optimal number of clusters is determined using the within-cluster sum of squares (WCSS) metric. 
The resulting clusters are visualized, and users can input their data for cluster prediction.

Code
------------------------

.. literalinclude:: C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/2. Unsupervised Learning/1. K-Means Clustering/Customer Spent Analysis .py
   :language: python


Optimal K Value Selection
--------------------------
.. image:: 1.1.png
   :alt: Finding Optimal K value
   :align: center

Result
-------

.. image:: 1.2.png
   :alt:  clustered result
   :align: center

Testing
--------
.. image:: 1.3.jpg
   :alt:  Testing
   :align: center