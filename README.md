# :sunrise_over_mountains: Barnes-Hut t-SNE using python
Comprehensive guidance for using Barnes-Hut t-SNE.

<p align="center">
  <img width="1000" height="700" src="/pic/bhtsne_resized.png" alt>
  <em>FIgure 1. Barnes-Hut t-SNE visualization using ImageNet dataset</em>
</p>

<p align="center">
  <img width="1000" height="700" src="/pic/tsne_resized.png" alt>
  <em>FIgure 2. t-SNE visualization using ImageNet dataset</em>
</p>


----
### :notes: Requirements
```shell
pip install -r requirements.txt
```


----
### :sunny: Try bh t-SNE

0. Just run `run.sh`
```shell
chmod 755 run.sh
./run.sh
```

or if you want to specific details, follow description.

1. Load numpy array and save text format.
```shell
python3 01_load_and_write.py
```

2. Compile Barnes-Hut t-SNE source.
```shell
g++ -O2 -g -Wall -Wno-sign-compare sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne
```

3. Run Barnes-Hut t-SNE to obtain reduced (projected) coordinates.
```shell
./bhtsne.py -v -r1 -i manifold_hist.txt -o manifold_hist.tsne.txt
```

4. Visualize and save png format.
```shell
python3 02_visualize_and_save.py
```

* cf) I already prepared sample ImageNet dataset in `data` folder.


----
### :umbrella: Try other methods (Random, PCA, Spectral, and t-SNE)

1. All you have to do is just execute below python file.
```shell
python3 03_compare_other_methods.py
```


----
### :cloud: Website
Author's website [[website]](https://lvdmaaten.github.io/tsne/)

Optimizing Barnes-Hut t-SNE (Microsoft research blog) [[website]](https://www.microsoft.com/en-us/research/blog/optimizing-barnes-hut-t-sne/)


----
### :palm_tree: Reference

Original paper [[paper]](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

----
### Milestone

- [x] Usage of bh t-SNE
- [x] Compare other methods

