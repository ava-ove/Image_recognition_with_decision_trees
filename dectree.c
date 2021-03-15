#include "dectree.h"

/**
 * This function takes in the name of the binary file containing the data and
 * loads it into memory. The binary file format consists of the following:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - 784 bytes : Image 1 data (28x28)
 *          ...
 *     -   1 byte  : Image N label
 *     - 784 bytes : Image N data (28x28)
 *
 * You can simply set the `sx` and `sy` values for all the images to 28, we
 * will not test this with different image sizes.
 */
Dataset *load_dataset(const char *filename) {
    // TODO: Allocate data, read image data / labels, return
  int N, error;
  FILE *file;

  Dataset *dataset_out = malloc(sizeof(Dataset));
  file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    exit(1);
  }
  //store number of images in N and in num_items 
  fread(&N, sizeof(int), 1, file);
  dataset_out->num_items = N;
  
  //malloc data set images and labels 
  dataset_out->images = malloc(sizeof(Image) * N);
  dataset_out->labels = malloc(N);  

  //loop num_imgs times
  for (int i=0; i<N; i++) {
    //image label
    fread(&(dataset_out->labels[i]), 1, 1, file);
    //setting sx and sy
    dataset_out->images[i].sx = 28;
    dataset_out->images[i].sy = 28;
    dataset_out->images[i].data = malloc(sizeof(unsigned char) * 28 * 28);
    //setting pixels of image[i]
    for (int j=0; j<28*28; j++) {
      fread(&(dataset_out->images[i].data[j]), 1, 1, file);
    }
  }
  
  error = fclose(file);
  if (error != 0) {
    fprintf(stderr, "Error: could not close file\n");
    exit(1);
  }

  return dataset_out;
}

/**
 * This function computes and returns the Gini impurity of the M images
 * (with their indices stored in the array) if they were to be split
 * based on the given pixel.
 * 
 * DO NOT CHANGE THIS FUNCTION; It is already implemented for you.
 */
double split_gini_impurity(Dataset *data, int M, int indices[M], int pixel) {
  int a_freq[10] = {0}, a_count = 0;
  int b_freq[10] = {0}, b_count = 0;
  for (int i = 0; i < M; i++) {
    int img_idx = indices[i];
    // The pixels are always either 0 or 255, but using < 128 for generality.
    if (data->images[img_idx].data[pixel] < 128)
      a_freq[data->labels[img_idx]]++, a_count++;
    else
      b_freq[data->labels[img_idx]]++, b_count++;
  }

  double a_gini = 0, b_gini = 0;
  for (int i = 0; i < 10; i++) {
    double a_i = ((double)a_freq[i]) / ((double)a_count);
    double b_i = ((double)b_freq[i]) / ((double)b_count);
    if (a_count) a_gini += a_i * (1 - a_i);
    if (b_count) b_gini += b_i * (1 - b_i);
  }
  // Weighted average of gini impurity of children
  return (a_gini * a_count + b_gini * b_count) / M;
}

/**
 * Given a subset of M images and their corresponding indices, find and return
 * the most frequent label and it's frequency. The last 2 parameters are
 * only for outputs.
 *
 * - The most frequent label should be stored in `*label`
 * - The frequency of this label should be stored in `*freq`
 * 
 * If multiple labels have the same maximal frequency, return the smallest one.
 */
void get_most_frequent(Dataset *data, int M, int indices[M], 
 int *label, int *freq) {
  // TODO: Set the correct values and return
  *label = 0;
  *freq = 0;
  int count_label[10] = {0}; //keep count of frequencies of labels 0 to 9
  for (int i=0; i<M; i++) {
    count_label[data->labels[indices[i]]]++;
  }

  //set freq to value of max item
  *freq = count_label[0]; //last element
  for (int i=0; i<10; i++) {
    //only change if you find larger frequency to get smallest label if same frequency
    if (count_label[i] > *freq) {
      *freq = count_label[i];
      *label = i; //set label as the index of that max item
    }
  }
  return; 
}

/**
 * Given a subset of M images and their corresponding indices, find and return
 * the best pixel to split the data based on. The best pixel is the one which
 * has the minimum Gini impurity as computed by `split_gini_impurity()`.
 * 
 * The return value should be a number between 0-783 (inclusive), representing
 *  the pixel the M images should be split based on.
 * 
 * If multiple pixels have the same minimal Gini impurity, return the smallest.
 */
int find_best_split(Dataset *data, int M, int indices[M]) {
    // TODO: Return the correct pixel 

  double gini_avgs[784];
  int min_i = 0;  
  //for each pixel call split_gini_impurity
  for (int i=0; i<784; i++) {
    gini_avgs[i] = split_gini_impurity(data, M, indices, i); 
  }

  //get index of minimum gini average and return
  // this is the pixel the M images should be split based on
  for (int i=0; i<784 ;i++) {
    if (gini_avgs[i] < gini_avgs[min_i]) {
      min_i = i;
    }
  }
  return min_i;
}

/**
 * This is the recursive function that creates the Decision tree. In each
 * recursive call, we only want to consider some subset of the entire dataset
 * corresponding to the node. To represent this, we pass in an array of indices
 * of these images in the dataset, along with it's length M. Be careful to
 * allocate this on the array for any recursive calls made, and free it when
 * you're done with creating the tree. Here, you need to:
 *
 *    - Compute ratio of most frequent image in current call, terminate if >95%
 *    - Find the best pixel to split on using `split_gini_impurity()`
 *    - Split the data, allocate corresponding arrays, and recurse
 *    - In either case, allocate a new node, set the correct values and return
 *       - If it's a child, set `classification`, and both children = NULL.
 *       - If it's not a child, set `pixel` and `left`/`right` (recursively). 
 */
DTNode *make_helper(Dataset *data, int M, int indices[M]) {
    // TODO: COnstruct and return tree
  int freq, label;
  DTNode *node = malloc(sizeof(DTNode)); //for either leaf or child node
  node->right = NULL;
  node->left = NULL;
  get_most_frequent(data, M, indices, &label, &freq);
  //reached a leaf, allocate a node and set classification to label
  if (M==0) return NULL; 
  if (((double)freq)/((double)M) >= TERMINATE_RATIO) {
    node->classification = label;
    node->pixel = -1;
    return node;
  }
  //not at a node so split data
  
  int bestSplitPixel = find_best_split(data, M, indices);
  //find number of images with index in indicies that have 0 at pixel bestSplitPixel
  int leftNumImgs = 0; 
  for (int i=0; i< M; i++) {
    if (data->images[indices[i]].data[bestSplitPixel] == 0)
      leftNumImgs++;
  }
  int rightNumImgs = M - leftNumImgs;
  //allocate arrays of images indicies with pixel 0 and 255 at bestSplitPixel 
  int *indiciesLeftChild = malloc(sizeof(int) * leftNumImgs);
  int *indiciesRightChild = malloc(sizeof(int) * rightNumImgs);

  int left_i = 0, right_i = 0; //to fill in arrays
  for (int i=0; i<M; i++) {
    int img_i = indices[i];
    if (data->images[img_i].data[bestSplitPixel] == 0)  {
      indiciesLeftChild[left_i] = img_i;
      left_i++;
    } else {
      indiciesRightChild[right_i] = img_i;
      right_i++;
    }
  }

  node->classification = -1;
  node->pixel = bestSplitPixel;

  node->right = make_helper(data, rightNumImgs, indiciesRightChild);
  free(indiciesRightChild);
  node->left = make_helper(data, leftNumImgs, indiciesLeftChild);
  free(indiciesLeftChild);
  

  return node;
}

/**
 * This is the function exposed to the user. All you should do here is set
 * up the `indices` array correctly for the entire dataset and call 
 * `make_helper()` with the correct parameters.
 */
DTNode *make_dec_tree(Dataset *data) {
  // TODO: Set up `indices` array, call `make_helper` and return tree.
  //     Make sure you free any data that is not needed anymore (hint) 
  int M = data->num_items;
  int *indices = malloc(sizeof(int) * M);  //indecies array
  for (int i=0; i<data->num_items; i++) 
    indices[i] = i;
  
  DTNode *tree = make_helper(data, M, indices);
  free(indices);
  //return makehelper and you pass in data and indecies
  return tree;
}

/**
 * Given a decision tree and an image to classify, return the predicted label.
 */
int dec_tree_classify(DTNode *root, Image *img) {
  //we are not at a leaf ie root is not classified
  if (root->classification == -1) {
    //pixel value is 0 so go left in tree
    if (img->data[root->pixel] == 0) 
      return dec_tree_classify(root->left, img);
    //pixel value is 255 so go right in tree
    return dec_tree_classify(root->right, img);
  } 

  //we are at a leaf ie root is classified
  //return correct label
  return root->classification;
}

/**
 * This function frees the Decision tree.
 */
void free_dec_tree(DTNode *node) {
  // Free the decision tree
  if (node == NULL) 
    return;
  free_dec_tree(node->left);
  free_dec_tree(node->right);

  free(node);
  return;
}

/**
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data) {
  // Free dataset (Same as A1)
  for (int i = 0; i < data->num_items; i++) 
    free(data->images[i].data);

  free(data->images);
  free(data->labels);
  free(data);
}