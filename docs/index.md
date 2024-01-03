# RAGatouille

_State-of-the-art document retrieval methods, in just a few lines of code._

---

Welcome to the documentation for RAGatouille. This section is mostly 🚧 Under Construction 🚧, but you can find a quick explainer for RAGatouille's philosophy and the strong performance of late-interaction models like ColBERT on this page, check out the [API reference for the main classes](https://ben.clavie.eu/ragatouille/api) as well as the [rough, ever-evolving roadmap](https://ben.clavie.eu/ragatouille/roadmap). The documentation will be actively updated in the next few weeks!

## Philosophy

RAGatouille's philosophy is two-fold.:

### Motivation/Aim

The first one is the __aim of this library__: bridging the seemingly growing gap between Information Retrieval litterature and everyday production retrieval uses. I'm not myself an IR researcher, and barely a practitioner: my own background is in NLP. However, the concept behind late-interaction retrievers and the many papers highlighting they're consistently better than dense embeddings on just about any zero-shot task have led me to them. On top of this, you don't need to use them zero-shot: they're very easy to adapt to new domains due to their bag-of-embeddings approach.

However, it's been a consistently hard sell, and starting to use ColBERT on real projects wasn't particularly smooth. IR is a field that is outwardly more _stern_ than NLP, and the barrier-to-entry is higher. A lot of the IR frameworks, like Terrier or Anserini, are absolutely fantastic, but they just don't fit into the pythonic day-to-day workflows we're used to.

For sentence transformers adoption, the aptly (re-)name SentenceTransformers library has been a boon. RAGatouille doesn't quite have the pretention to be that, but it aims to help democratise the easy training and use ColBERT and pals. To do so, we also take an approach of avoiding re-implementing whenever possible, to speed up iteration.

If a paper has open-sourced its code, our goal is for RAGatouille to be a gateway to it, rather than a complete replacement, whenever possible! (_This might change in the future as the library grows_) Moreover, this is a mutually beneficial parasitic relationships, as the early development of this lib has already resulted in a few upstreamed fixes on the main ColBERT repo!

### Code Philosophy

The actual programming philosophy is fairly simple, as stated on the [github README](https://github.com/bclavie/RAGatouille):

- Strong, but parameterable defaults: you should be able to get started with just a few lines of code and still leverage the full power of ColBERT, and you should be able to tweak any relevant parameter if you need to!
- Powerful yet simple re-usable components under-the-hood: any part of the library should be usable stand-alone. You can use our DataProcessor or our negative miners outside of `RAGPretrainedModel` and `RAGTrainer`, and you can even write your own negative miner and use it in the pipeline if you want to!

In practice, this manifests by the fact that RAGPretrainedModel and RAGTrainer should ultimately be _all you need_ to leverage the power of ColBERT in your pipelines. Everything that gets added to RAGatouille should be workable into these two classes, who are really just interfaces between the underlying models and processors.

However, re-usable components is another very important aspect. RAGatouille aims to be built in such a way that every core component, such as our negative miners (`SimpleMiner` for dense retrieval at the moment) or data processors should be usable outside the main classes, if you so desire. If you're a seasoned ColBERT afficionado, nothing should stop you from importing `TrainingDataProcessor` to streamline processing and exporting triplets!

Finally, there's a third point that's fairly important:

- __Don't re-invent the wheel.__

If a component needs to do something, we won't seek to do it our way, we'll seek to do it the way people already do it. This means using `LlamaIndex` to chunk documents, `instructor` and `pydantic` to constrain OpenAI calls, or `DSPy` whenever we need more complex LLM-based components!

## Late-Interaction

### tl;dr

So, why is late-interaction so good? Why should you use RAGatouille/ColBERT?

The underlying concept is simple. Quickly put, I like to explain ColBERT as a `bag-of-embeddings` approach, as this makes it immediately obvious how and why ColBERT works to NLP practitioners:

- Just like bag-of-words, it works on small information units, and represents a document as the sum of them
- Just like embeddings, it works on the semantic level: the actual way something is phrased doesn't matter, the model learns __meaning__.

### longer, might read

A full blog post with more detail about this is coming (soon™), but for now, here's a quick explainer:

Take it this way, the existing widely-used retrieval approaches, and a quick overview of their pros and cons:

##### BM25/Keyword-based Sparse Retrieval

➕ Fast  
➕ Consistent performance  
➕ No real training required  
➕ Intuitive  
➖ Requires exact matches  
➖ Does not leverage any semantic information, and thus hits __a hard performance ceiling__  

##### Cross-Encoders

➕ Very strong performance  
➕ Leverages semantic information to a large extent ("understands" negative form so that "I love apples" and "I hate apples" are not similar, etc...)  
➖ Major scalability issues: can only retrieve scores by running the model to compare a query to every single document in the corpus.  

##### Dense Retrieval/Embeddings

➕ Fast  
➕ Decent performance overall, once pre-trained  
➕ Leverages semantic information...  
➖ ... but not constrastive information (e.g. "I love apples" and "I hate apples" will have a high similarity score.)  
➖ Fine-tuning can be finnicky  
➖ Requires either billions of parameters (e5-mistral) or billions of pre-training examples to reach top performance  
➖ __Often generalises poorly__  

### Generalisation

This last point is particularly important. __Generalisation__ is what you want, because the documents that you're trying to retrieve for your users, as well as the way that they phrase their queries, are __not the ones present in academic datasets__.

Strong performance on academic benchmark is a solid signal to predict how well a model will perform, but it's far from the only ones. Single-vector embeddings approach, or _dense retrieval_ methods, often do well on benchmarks, as they're trained specifically for them. However, the IR litterature has shown many times that these models often genralise worse than other approaches.

This is not a slight on them, it's actually very logical! If you think about it:

- A single-vector embedding is **the representation of a sentence or document into a very small vector space, with at most 1024 dimensions**.
- In retrieval settings, the **same model must also be able to create similar representations for very short query and long documents, to be able to retrieve them**.
- Then, **these vectors must be able to represent your documents and your users' query, that it has never seen, in the same way that it has learned to represent its training data**. 
- And finally, **it must be able to encode all possible information contained in a document or in a query, so that it may be able to find a relevant document no matter how a question is phrased**

The fact that dense embeddings perform well in these circumstances is very impressive! But sadly, embedding all this information into just a thousand dimension isn't a proble, that has been cracked yet. 

### Bag-of-Embeddings: the Late Interaction trick

Alleviating this is where late-interaction comes in.

ColBERT does not represent documents into a single vector. In fact, ColBERT, at its core, is basically a **keyword-based approach**.

But why does it perform so well, then, when we've established that keyword matchers have a hard ceiling?

Because ColBERT is a **semantic keyword matcher**. It leverages the power of strong encoder models, like BERT, to break down each document into a bag of **contextualised units of information**. 

When a document is embedded by ColBERT, it isn't represented as a document, but as the sum of its parts.

This fundamentally changes the nature of training our model, to a much easier task: it doesn't need to cram every possible meaning into a single vector, it just needs to capture the meaning of a few tokens at a time. When you do this, it doesn't really matter how you phrase something at retrieval time: the likelihood that the model is able to relate the way you mention a topic to the way a document discusses it is considerably higher. This is quite intuitively because the model has so much more space to store information on individual topics! Additionally, because this allows us to create smaller vectors for individual information units, they become very compressable, which means our indexes don't balloon up size.