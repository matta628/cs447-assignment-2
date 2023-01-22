#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <algorithm>
#include <vector>
#include <queue>
#include <iostream>
#include <utility>

#include "KDtree.hpp"

using namespace std;

struct BuildParams{
	float* arr;
	int n_pts;
	int dim;
	int k;
	int first_run;
	KDtree::Node *node;
};

int compare(const void *o1, const void*o2, void *vp){
	int dim = *(int *)vp;
	if (*((float *)o1 + dim) < *((float *)o2 + dim))
		return -1;
	else if (*((float *)o1 + dim) > *((float *)o2 + dim))
		return 1;
	return 0;
}

extern "C" void *build(void *vp){
	BuildParams *p = (BuildParams *)vp;
	queue<BuildParams *> work;
	work.push(p);

	BuildParams *curr;
	while (!work.empty()){
		//get next task from queue
		curr = work.front();
		work.pop();

		float* arr = curr->arr;
		int n_pts = curr->n_pts;
		int dim = curr->node->dim;
		int k = curr->k;

		float median;
		if (curr->first_run != 1){ //if not firstRun
			//choose n_samples random points
			int n_samples = 999;
			int sample_sz = n_samples*k; //flattened array of samples size
			float *sample = new float[sample_sz]; //flattened array of samples
			unsigned int seed = 1234; 
			for (int i = 0; i < sample_sz; i += k){
				int rand_ind = rand_r(&seed) % n_pts;
				for (int j = 0; j < k; j++){
					sample[i + j] = arr[rand_ind*k + j];
				}
			}

			//sort sample according to specified dimension
			qsort_r((void *)sample, n_samples, k*sizeof(float), compare, (void*)&dim);
		
			//find median point from sample 
			median = sample[(n_samples/2)*k + dim];
		
			//delete sample
			delete[] sample;
		
			//set node's axis to the median from the sample
			curr->node->axis = median;
		}
		else{
			//cout << "first run! " <<endl;
			curr->first_run = 0; //false
			median = curr->node->axis; //median already calculated
		}

		//two vectors
		vector<float> less; //or equal to
		vector<float> more;
	
		//add points to corresponding vectors
		for (int i = 0; i < n_pts; i++){
			if (arr[i*k + dim] <= median)
				for (int j = 0; j < k; j++)
					less.push_back(arr[i*k + j]);
			else
				for (int j = 0; j < k; j++)
					more.push_back(arr[i*k + j]);
		}

		//if median is smallest or largest
		if(less.size() == 0 | more.size() == 0){
			less.clear();
			more.clear();
			for (int i = 0; i < n_pts; i++){
				for (int j = 0; j < k; j++){
					if (i < n_pts/2)
						less.push_back(arr[i*k + j]);
					else
						more.push_back(arr[i*k + j]);
				}
			}
		}

		//two new flattened k-d arrays
		float * leftArr = new float[less.size()];
		float * rightArr = new float[more.size()];
		//copy vectors to arrays on heap
		memcpy((void *)leftArr, (const void*)&less[0], less.size() * sizeof(float));
		memcpy((void *)rightArr, (const void*)&more[0], more.size() * sizeof(float));

		if (n_pts <= 16){//no more splitting needed
			//create two leaf nodes..
			curr->node->left =
				new KDtree::Leaf(curr->node->axis, curr->node->dim, leftArr, less.size()/k);
			curr->node->right =
				new KDtree::Leaf(curr->node->axis, curr->node->dim, rightArr, more.size()/k);
		}
		else { //further splitting required
			//create new left and right nodes with correct dim
			int next_dim = (dim+1)%k;
			KDtree::Node* left = new KDtree::Node(0, next_dim);
			curr->node->left = left;
			KDtree::Node* right = new KDtree::Node(0, next_dim);
			curr->node->right = right;

			BuildParams *l_params = new BuildParams();
			l_params->node = left;
			l_params->arr = leftArr;
			l_params->n_pts = less.size()/k;
			l_params->dim = dim;
			l_params->k = k;
			l_params->first_run = 0;
			work.push(l_params);

			BuildParams *r_params = new BuildParams();
			r_params->node = right;
			r_params->arr = rightArr;
			r_params->n_pts = more.size()/k;
			r_params->dim = dim;
			r_params->k = k;
			r_params->first_run = 0;
			work.push(r_params);
		}

		//delete old array
		delete[] arr;
		
		//delete buildparams obj after use except for original one
		if (curr != p)
			delete curr;
	}

	return vp;
}

KDtree::KDtree(int dims, float *pts, int n_pts, float *queries, int n_queries, int n_cores){
	this->k = dims;
	this->points = pts; //data is a flattened array of k-D points
	this->n_points = n_pts; //number of points
	this->n_threads = n_cores;
	this->queries = queries;
	this->n_queries = n_queries;
	assert(n_points > 0);
	assert(root == nullptr);
	
	int t = n_threads;
	this->max_depth = 0;
	// use max power of 2 number of threads for construction..
	// 2,3 cores -> 2 threads
	// 4,5,6,7 cores -> 4 threads
	while (t >= 2){
		this->max_depth++;
		t >>= 1; 
	}
	//set t to 2^max_depth
	t = 1 << max_depth;
	
	//split kd tree construction into arr, arr_size pairs
	vector<BuildParams *> subtrees;
	this->root = splitBuild(root, points, n_points, 0, 0, subtrees);

	//create t threads
	pthread_t tids[32];

	for (int i = 0; i < t; i++){
		pthread_t tid;
		void *vp = (void *)subtrees[i];
		pthread_create(&tid, NULL, build, vp);
		tids[i] = tid;
	}

	//join t threads
	for (int i = 0; i < t; i++){
		void *vp;
		pthread_join(tids[i], &vp);
		delete (BuildParams*)vp;
	}
	
}

KDtree::Node* KDtree::splitBuild(Node *node, float* arr, int n_pts, int dim, int depth, vector<BuildParams *>& subtrees){
	//choose n_samples random points
	int n_samples = 999;
	int sample_sz = n_samples*k; //flattened array of samples size
	float *sample = new float[sample_sz]; //flattened array of samples
	unsigned int seed = 1234; 
	for (int i = 0; i < sample_sz; i += k){
		int rand_ind = rand_r(&seed) % n_pts;
		for (int j = 0; j < k; j++){
			sample[i + j] = arr[rand_ind*k + j];
		}
	}

	//sort sample according to specified dimension
	qsort_r((void *)sample, n_samples, k*sizeof(float), compare, (void*)&dim);

	//find median point from sample 
	float median = sample[(n_samples/2)*k + dim];

	//delete sample
	delete[] sample;

	//create new node using median value w respect to dim
	node = new Node(median, dim);

	//no more splitting
	if (depth == max_depth){

		//add buildparams struct using created node at max depth to subtrees
		BuildParams * params = new BuildParams();
		params->arr = arr;
		if (max_depth == 0){ //if only one core don't send original array of pts
			int arr_sz = n_pts*k;
			float *arr_copy = new float[arr_sz];
			memcpy((void *)arr_copy, (const void *)arr, arr_sz * sizeof(float));
			params->arr = arr_copy;
		}
		params->n_pts = n_pts;
		params->dim = dim;
		params->k = k;
		params->node = node;
		params->first_run = 1; //true
		subtrees.push_back(params);
	}
	else{ //keep splitting
		//two vectors
		vector<float> less; //or equal to
		vector<float> more;
		
		//add points to corresponding vectors
		for (int i = 0; i < n_pts; i++){
			if (arr[i*k + dim] <= median)
				for (int j = 0; j < k; j++)
					less.push_back(arr[i*k + j]);
			else
				for (int j = 0; j < k; j++)
					more.push_back(arr[i*k + j]);
		}

		//if arr != this->points, then delete arr
		if (arr != points)
			delete[] arr;
		
		//two new arrays
		float * leftArr = new float[less.size()];
		float * rightArr = new float[more.size()];
	
		//copy vectors to arrays on heap
		memcpy((void *)leftArr, &less[0], less.size() * sizeof(float));
		memcpy((void *)rightArr, &more[0], more.size() * sizeof(float));

		//call build on left and right subtrees...
		node->left = splitBuild(node->left, leftArr, less.size()/k, (dim+1) % k, depth+1, subtrees);
		node->right = splitBuild(node->right, rightArr, more.size()/k, (dim+1) % k, depth+1, subtrees);
	}
	
	return node;
}

struct QueryParams{
	float* arr;
	int n_q;
	int dims;
	int nn;
	KDtree::Node *root;
};

struct ComparePoints{
	bool operator() (const pair<float*,float> a, const pair<float*,float> b) const{
		return a.second < b.second;
	}
};

void nearestNeighbors(float *query, KDtree::Node *node, int dim, int nn, int k,
							 priority_queue<pair<float*,float>, vector<pair<float*,float>>, ComparePoints>& neighbors){
	//check if leaf node
	if (node->right == nullptr && node->left == nullptr){
		KDtree::Leaf *leaf = (KDtree::Leaf *)node;
		float *arr = leaf->pts;
		int n_pts = leaf->n_pts;

		//go through all points at leaf
		for (int i = 0; i < n_pts; i+=k){
			float dist = 0;
			for (int j = 0; j < k; j++){ //calc dist from each point to query
				float diff = arr[i+j];
				diff -= query[j];
				//float diff = arr[i+j] - query[j];
				dist += diff*diff;
			}
			if (neighbors.size() < (unsigned long)nn){ //add all points to pq
				pair<float*,float> point (&arr[i], dist);
				neighbors.push(point);
			}
			else{ // neighbors.size == nn
				if (dist < neighbors.top().second){ //curr_dist < farthest neighbor
					neighbors.pop();
					pair<float*,float> point (&arr[i], dist);
					neighbors.push(point);
				}
			}
		}
		return;
	}

	if (query[dim] <= node->axis){ //check left subtree than right if needed
		nearestNeighbors(query, node->left, (dim+1)%k, nn, k, neighbors); 
		//if neighbors doesnt have nn points or query is closer to axis than to current farthest neighbor
		if (neighbors.size() < (unsigned long)nn || abs(query[dim] - node->axis) < neighbors.top().second)
			nearestNeighbors(query, node->right, (dim+1)%k, nn, k, neighbors);
	}
	else{ //query[dim] > node->axis
		nearestNeighbors(query, node->right, (dim+1)%k, nn, k, neighbors);
		//if neighbors doesnt have nn points or query is closer to axis than to current farthest neighbor
		if (neighbors.size() < (unsigned long)nn || abs(query[dim] - node->axis) < neighbors.top().second)			
			nearestNeighbors(query, node->left, (dim+1)%k, nn, k, neighbors);
	}
	return;
}

extern "C" void *query(void *vp){
	QueryParams *p = (QueryParams *)vp;
	float *queries = p->arr;
	int n_queries = p->n_q;
	int dims = p->dims;
	int nn = p->nn;
	KDtree::Node *root = p->root;
	delete p;
	
	float* results = new float[n_queries*nn*dims];
	for (int q = 0; q < n_queries; q++){
		priority_queue<pair<float*,float>, vector<pair<float*,float>>, ComparePoints> neighbors;
		nearestNeighbors(&queries[q*dims], root, 0, nn, dims, neighbors);

		vector<float> nn_closest(nn*dims); 
		//rearrange nn nearest neighbors from closest to farthest
		for (int i = dims*(nn-1); i >= 0; i-=dims){
			float *pt = neighbors.top().first;
			for (int j = 0; j < dims; j++){
				nn_closest[i+j] = pt[j];
			}
			neighbors.pop();
		}

		memcpy((void *)&results[q*nn*dims], (const void *)&nn_closest[0], nn*dims*sizeof(float));
	}
	return (void *)results;
}

float *KDtree::processQueries(int nn){
	float *results = new float[n_queries*k*nn];

	//create t threads
	pthread_t tids[32];
	int t = this->n_threads;
	for (int i = 0; i < t; i++){
		pthread_t tid;
		int n_q = n_queries / t;
		float *arr = &queries[i*n_q*k]; 
		if (i == t-1)
			n_q += n_queries % t;
		QueryParams *params = new QueryParams();
		params->arr = arr;
		params->n_q = n_q;
		params->dims = k;
		params->nn = nn;
		params->root = root;
		void *vp = (void *)params;
		pthread_create(&tid, NULL, query, vp);
		tids[i] = tid;
	}

	//join t threads
	for (int i = 0; i < t; i++){
		void *vp;
		pthread_join(tids[i], &vp);
		//can the threads write to the results directly?
		int n_q = n_queries/t;
		int offset = i*n_q*nn*k;
		if (i == t-1)
			n_q += n_queries % t;
		memcpy((void *)&results[offset], (const void *)vp, n_q*nn*k*sizeof(float));
		delete[] (float *)vp;
	}
	return results;
}

KDtree::~KDtree(){
	delete root;
}

KDtree::Node::~Node(){
	delete left;
	delete right;
}

KDtree::Leaf::~Leaf(){
	delete[] pts;
}

