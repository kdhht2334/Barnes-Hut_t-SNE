echo "First, load numpy array and save text format."
python3 01_load_and_write.py

echo "Second, compile bh T-SNE source."
g++ -O2 -g -Wall -Wno-sign-compare sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne

echo "Third, run bh T-SNE to make projected coordinates."
./bhtsne.py -v -r1 -i manifold_hist.txt -o manifold_hist.tsne.txt

echo "Last, visualize and save png format."
python3 02_visualize_and_save.py

echo "Finish!"
