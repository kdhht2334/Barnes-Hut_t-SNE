# Barnes-Hut t-SNE using python
Comprehensive guidance for using Barnes-Hut t-SNE.

<p align="center">
  <img width="500" height="250" src="/pic/bhtsne_resized.png">
</p>

----
#### Usage

1. First, load numpy array and save text format.
```shell
python3 01_load_and_write.py
```

2. Second, compile Barnes-Hut t-SNE source.
```shell
g++ -O2 -g -Wall -Wno-sign-compare sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne
```

3. Third, run Barnes-Hut t-SNE to obtain reduced (projected) coordinates.
```shell
./bhtsne.py -v -r1 -i manifold_hist.txt -o manifold_hist.tsne.txt
```

4. Last, visualize and save png format.
```shell
python3 02_visualize_and_save.py
```

* cf) I already prepared sample ImageNet dataset in `data` folder.


----
#### Website
Author's website [[website]](https://lvdmaaten.github.io/tsne/)

Optimizing Barnes-Hut t-SNE (Microsoft research blog) [[website]](https://www.microsoft.com/en-us/research/blog/optimizing-barnes-hut-t-sne/)


----
#### Reference

Original paper [[paper]](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

