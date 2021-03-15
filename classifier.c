#include "dectree.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets
//
// Or, to compile manually, use:
//    gcc -o classifier dectree.c classifier.c -g -Wall -std=c99 -lm
//
// Running decision tree generation / validation:
//    ./classifier datasets/training_data.bin datasets/testing_data.bin

/*****************************************************************************/
/* Do not add anything outside the main function here. Any core logic other  */
/* than what is described below should go in `dectree.c`. You've been warned!*/
/*****************************************************************************/

/**
 * main() takes in 2 command line arguments:
 *    - training_data: A binary file containing training image / label data
 *    - testing_data: A binary file containing testing image / label data
 *
 * You need to do the following:
 *    - Parse the command line arguments, call `load_dataset()` appropriately.
 *    - Call `make_dec_tree()` to build the decision tree with training data
 *    - For each test image, call `dec_tree_classify()` and compare the real 
 *        label with the predicted label
 *    - Print out (only) one integer to stdout representing the number of 
 *        test images that were correctly classified.
 *    - Free all the data allocated and exit.
 * 
 *  (You should for the most part be able to re-use code from A1 for this)
 */
int main(int argc, char *argv[]) {
 // Handle command line arguments
  int LENGTH = 256; 
  char training_data[LENGTH];
  char testing_data[LENGTH];

  if (argc != 3) {
  	fprintf(stderr, "Incorrect number of arguments given\n");
  	exit(1);
  }
  strncpy(training_data, argv[1], strlen(argv[1]) + 1);
  strncpy(testing_data, argv[2], strlen(argv[2]) + 1);

  // Compute the total number of correct predictions
  int total_correct = 0;
  int predicted_label;
  Dataset *training_data_set = load_dataset(training_data);
  Dataset *testing_data_set = load_dataset(testing_data);

  DTNode *treeroot = make_dec_tree(training_data_set);
  for (int i=0; i<testing_data_set->num_items; i++) {
  	predicted_label = dec_tree_classify(treeroot, &(testing_data_set->images[i]));
  	if (predicted_label == testing_data_set->labels[i])
  		total_correct++;
  }

  // Print out answer
  printf("%d\n", total_correct);

  // free all space
  free_dataset(training_data_set);
  free_dataset(testing_data_set);
  free_dec_tree(treeroot);

  return 0;
}