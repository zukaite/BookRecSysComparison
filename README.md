
# Benchmarking State-of-the-Art Recommender Models

## Project Overview
This project is an exploration into the world of recommender systems, focusing on benchmarking state-of-the-art models using the Goodreads book review dataset. It represents a personal journey into RecSystems, offering insights into different methodologies and their effectiveness in recommending books.

## Objective
The primary objective was to benchmark and compare four prominent recommender models: SVD, SVAE, NGCF, and LightGCN. This comparison not only serves as a study of their performances but also as an educational experience in understanding the nuances of each model.

## Models Explored

1. **SVD (Singular Value Decomposition)**: A foundational technique in recommendation systems. It's important to note that in practical, large-scale systems, alternatives like ALS are often preferred due to their scalability and efficiency with large datasets. My exploration of SVD serves as a conceptual baseline, offering a clear view of the fundamentals of matrix factorization before advancing to more complex and scalable models.

2. **SVAE (Variational Autoencoders for Collaborative Filtering)**: Integrating deep learning into collaborative filtering.
3. **NGCF (Neural Graph Collaborative Filtering)**: Employing graph neural networks for capturing user-item interactions.
4. **LightGCN (Light Graph Convolution Network)**: Simplifying graph convolutional networks for efficient recommendation.


## Comparative Results
The models were evaluated on metrics such as Precision@k, Recall@k, MAP, and NDCG (k=10).

| Model    | Precision@k | Recall@k | MAP      | NDCG     |
| -------- | ----------- | -------- | -------- | -------- |
| SVD      | 0.007586    | 0.004831 | 0.001789 | 0.008222 |
| SVAE     | 0.207000    | 0.071519 | 0.041507 | 0.216293 |
| NGCF     | 0.153611    | 0.111269 | 0.038892 | 0.151472 |
| LightGCN | 0.200073    | 0.148815 | 0.080734 | 0.240993 |

## Insights from Model Comparisons

This project not only served as a benchmarking exercise but also as a learning platform to delve deep into the functionalities and effectiveness of different recommender systems. Here's what was uncovered:

- **SVD's Simplicity vs. Complexity**: SVD, with its lower scores, highlights a critical trade-off between simplicity and the need for more complex models in handling diverse datasets.
- **Deep Learning's Edge in SVAE**: The significant leap in performance with SVAE demonstrates the power of deep learning in capturing complex, non-linear user-item interactions that traditional matrix factorization methods might miss.
- **Graph-based Insights in NGCF**: NGCF's performance indicates the value of incorporating user-item interaction graphs, offering a more interconnected perspective compared to traditional methods.
- **Efficiency of LightGCN**: LightGCN's balanced approach between complexity and performance, as evidenced by its highest NDCG score, showcases the effectiveness of graph convolutional networks in recommendation systems.

These results underline the importance of context when choosing the right model. While one model might excel in a particular scenario, another might be more suited to a different set of data or user behavior patterns.


## Setup
0. Python 3.10.9 was used for this project.
1. Clone the repository to your local machine.
2. Install the required dependencies using the `requirements.txt` file: 
   `pip install -r requirements.txt`


## Contact
Feel free to reach out for discussions or queries related to this project.

## References

- 'Better Than Netflix Movie Recommender' [GitHub Repository](https://github.com/nlp-nathan/better_than_netflix_movie_recommender/tree/master)

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. [Link](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

- Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008). Large-Scale Parallel Collaborative Filtering for the Netflix Prize. [Link](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf)

- Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, & Tat-Seng Chua (2019). Neural Graph Collaborative Filtering. [arXiv:1905.08108](https://arxiv.org/abs/1905.08108)

- Kilol Gupta, Mukund Y. Raghuprasad, Pankhuri Kumar (2018). A Hybrid Variational Autoencoder for Collaborative Filtering. [arXiv:1808.01006](https://arxiv.org/pdf/1808.01006.pdf)

- Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang & Meng Wang (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)

## License
The project is open-sourced under [MIT License](LICENSE).

---

*Happy Learning and Exploring Recommender Systems!*