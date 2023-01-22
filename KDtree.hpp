#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <queue>

struct BuildParams;
struct ComparePoints;

class KDtree{
	int k;
	float *points;
	int n_points;
	float *queries;
	int n_queries;
	int n_threads;
	int max_depth;

  public:
	class Node{
	  public:
		Node(float data, int d): axis(data), dim(d){}
		float axis;
		int dim;
		Node* left = nullptr;
		Node* right = nullptr;
		virtual ~Node();
	};
	class Leaf : public Node {
	  public:
		Leaf(float median, int dim, float * data, int num_pts):
			Node(median, dim), pts(data), n_pts(num_pts) {}
		float * pts;
		int n_pts;
		virtual ~Leaf();
	};

	KDtree(int, float*, int, float*,int, int);
	Node* splitBuild(Node*, float*, int, int, int, std::vector<BuildParams *>&); 
	void print();
	float* processQueries(int); 
	void nearestNeighbors(float*, Node*, int, int, int,
						 std::priority_queue<std::pair<float*,float>, std::vector<std::pair<float*,float>>, ComparePoints>&);
	~KDtree();
  private:
	Node *root = nullptr;
};
#endif
