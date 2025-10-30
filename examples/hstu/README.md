# Examples: to demonstrate how to do training and inference generative recommendation models

## Generative Recommender Introduction
Meta's paper ["Actions Speak Louder Than Words"](https://arxiv.org/abs/2402.17152) introduces a novel paradigm for recommendation systems called **Generative Recommenders(GRs)**, which reformulates recommendation tasks as generative modeling problems. The work introduced Hierarchical Sequential Transduction Units (HSTU), a novel architecture designed to handle high-cardinality, non-stationary data streams in large-scale recommendation systems. HSTU enables both retrieval and ranking tasks. As noted in the paper, “HSTU-based GRs, with 1.5 trillion parameters, improve metrics in online A/B tests by 12.4% and have been deployed on multiple surfaces of a large internet platform with billions of users.”

In this example, we introduce the model architecture, training, and inference processes of HSTU. For more details, refer to the [training](./training/) and [inference](./inference/) entry folders, which include comprehensive guides and benchmark results.

## Ranking Model Introduction
The model structure of the generative ranking model can be depicted by the following picture.
![ranking model structure](./figs/ranking_model_structure.png)

### Input
The input to the HSTU model consists solely of pure categorical features, and it does not accommodate numerical features. The model supports three types of tokens:
* Contextual Tokens: Represent the user side info.
* Item Tokens: Represent the items being recommended.
* Action Tokens: Optional. Represent user actions associated with these items. Please note that if a user has multiple actions associated with a single item token, these actions must be merged into a single token during data preprocessing. For further details, please refer to [the related issue](https://github.com/facebookresearch/generative-recommenders/issues/114).

It is crucial that the number of item tokens matches the number of action tokens. This alignment ensures that each item can be effectively paired with its corresponding user action, as the paper said.

### Embedding Table
The embedding mechanism includes three types of distinct tables:
* Contextual Embedding Table: Corresponds to contextual tokens.
* Item Embedding Table: Corresponds to item tokens.
* Action Embedding Table: Corresponds to action tokens if provided.

### HSTU Block
The HSTU block is a core component of the architecture, which modifies traditional attention mechanisms to effectively handle large, non-stationary vocabularies typical in recommendation systems. 
* **Preprocessing**: After retrieving the embedding vectors from the tables, the HSTU preprocessing stage follows. If action embeddings are provided, the model interleaves the item and action embedding vectors. It then concatenates the contextual embeddings with the interleaved item and action embeddings, ensuring that each sample starts with contextual embeddings followed by item and action sequence pairs. Finally, the model applies position encoding.

* **Postprocessing**: If candidate items are specified, the model predicts only these candidates by filtering candidate item embeddings in the postprocessing. Otherwise, all item embeddings will be selected to be used for prediction.

### Prediction Head
The prediction head of the HSTU model employs a MLP network structure, enabling multi-task predictions. 

## Running the examples

* [HSTU training example](./training/)
* [HSTU inference example](./inference/)

# Acknowledgements

We would like to thank Yueming Wang (yuemingw@meta.com) and Jiaqi Zhai(jiaqiz@meta.com) for their guidance and assistance with the paper Action Speaks Louder Than Words during our efforts to understand the algorithm and reproduce the results. We also extend our gratitude to all the authors of the paper for their contributions and guidance. In addition, we would like to express special thanks to developers of [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) that we have referenced. 
