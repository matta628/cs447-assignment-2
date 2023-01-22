#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cstdint>

#include <x86intrin.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#include "KDtree.hpp"

using namespace std;

inline auto start_tsc(){
	_mm_lfence();
	auto tsc = __rdtsc();
	_mm_lfence();
	return tsc;
}

inline auto stop_tsc(){
	unsigned int aux;
	auto tsc = __rdtscp(&aux);
	_mm_lfence();
	return tsc;
}

int main(int argc, char **argv){
	if (argc != 5) return -1;
	int n_cores = atoi(argv[1]);
	char *train_file = argv[2];
	char *query_file = argv[3];
	char *result_file = argv[4];

	uint64_t train_fid; //training file ID
	uint64_t query_fid; //query file ID
	uint64_t result_fid = 0;
	int rdnm_fd = open("/dev/urandom", O_RDONLY);
	if (rdnm_fd < 0){
		cout << "problem opening dev/urandom" << endl;
		return -1;
	}
	int rv = read(rdnm_fd, (void *)&result_fid, sizeof(result_fid)); assert(rv >= 0);
	uint64_t n_points; //number of training n_points
	uint64_t n_queries; //number of n_queries
	uint64_t dims; //number of dimensions
	uint64_t nn; //number of neighbors for each query

	void *tp;
	void *qp;
	{
		//training file
		int fd = open(train_file, O_RDONLY);
		if (fd < 0){
			cout << "couldn't open training file" << endl;
			return -1;
		}
	
		struct stat sb;
		int rv = fstat(fd, &sb); assert(rv==0);
		assert(sb.st_size % sizeof(double) == 0);

		tp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
		if (tp == MAP_FAILED){
			cout << "training file mmap failed" << endl;
			return -1;
		}

		rv = madvise(tp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv==0);
		rv = close(fd); assert(rv==0);
		n_points = sb.st_size/sizeof(double);

		//query file
		fd = open(query_file, O_RDONLY);
		if (fd < 0){
			cout << "couldn't open query file" << endl;
			return -1;
		}

		rv = fstat(fd, &sb); assert(rv==0);
		assert(sb.st_size % sizeof(double) == 0);

		qp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
		if (qp == MAP_FAILED){
			cout << "query file mmap failed" << endl;
			return -1;
		}

		rv = madvise(qp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv==0);
		rv = close(fd); assert(rv == 0);
		n_queries = sb.st_size/sizeof(double);
	}

	//TRAINING FILE SETUP
	char * train1B = (char *)tp; //1 byte
	int n = strnlen(train1B, 8);
	assert(string(train1B, n) == "TRAINING");
	uint64_t *train8B = (uint64_t *)tp; //8 bytes
	train8B++; //skip file type string
	train_fid = *train8B; train8B++;
	n_points = *train8B; train8B++;
	dims = *train8B; train8B++;

	//store points in array
	float *points = new float[n_points*dims];
	float *train4B = (float *)train8B; //4 bytes 
	for (uint64_t i = 0; i < n_points; i++){
		for (uint64_t j = 0; j < dims; j++){
			points[i*dims + j] = *train4B; train4B++;
		}
	}
	
	//QUERY FILE SETUP
	char * query1B = (char *)qp;
	n = strnlen(query1B, 8);
	assert(string(query1B, n) == "QUERY");
	uint64_t *query8B = (uint64_t *) qp;
	query8B++; //skip file type string
	query_fid = *query8B; query8B++;
	n_queries = *query8B; query8B++;
	uint64_t queryDims = *query8B; query8B++;
	assert(queryDims == dims);
	nn = *(uint64_t *)query8B; query8B++;

	//store queries in array
	float *query4B = (float *)query8B;
	float *queries = new float[n_queries*dims];
	for (uint64_t i = 0; i < n_queries; i++){
	 	for (uint64_t j = 0; j < dims; j++){
			queries[i*dims + j] = *query4B; query4B++;
		}
	}

	//calc clock frequency
	auto zero = start_tsc();
	sleep(1);
	auto one = stop_tsc();
	double freq = one - zero;

	//build kdtree using points
	auto begin = start_tsc();
	KDtree kdtree = KDtree(dims, points, n_points, queries, n_queries, n_cores);
	auto end = start_tsc();
	double build_time = (end-begin)/freq;
	printf("build time: %fs\n", build_time);
	
	//delete points
	delete[] points;

	//process all queries and save to results
	begin = start_tsc();
	float * results = kdtree.processQueries(nn);
	end = stop_tsc();
	double query_time = (end-begin)/freq;
	printf("query time: %fs\n", query_time);

	//delete queries
	delete[] queries;

	//write results to file
	char *rp;
	{
		int fd = open(result_file, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
		if (fd < 0){
			cout << "couldn't open result file" << endl;
			fprintf(stderr, "Error is %s\n", strerror(errno));
			return -1;
		}
		int size = 7*8 + n_queries*nn*dims*sizeof(float);
		ftruncate(fd, size);
		
		rp = (char *) mmap(nullptr, size, PROT_WRITE, MAP_SHARED, fd, 0);
		if (rp == MAP_FAILED){
			cout << "result file mmap failed" << endl;
			fprintf(stderr, "Error is %s\n", strerror(errno));
			return -1;
		}
		auto rv = close(fd); assert(rv==0);
	}

	//write result file header info
	memcpy(rp, (const void *)"RESULT", 6);
	rp += 8;
	memcpy(rp, (const void *)&train_fid, sizeof(train_fid));
	rp += 8;
	memcpy(rp, (const void *)&query_fid, sizeof(query_fid));
	rp += 8;
	memcpy(rp, (const void *)&result_fid, sizeof(result_fid));
	rp += 8;
	memcpy(rp, (const void *)&n_queries, sizeof(n_queries));
	rp += 8;
	memcpy(rp, (const void *)&dims, sizeof(dims));
	rp += 8;
	memcpy(rp, (const void *)&nn, sizeof(nn));
	rp += 8;
	memcpy(rp, (const void *)results, n_queries*dims*nn*sizeof(float));
	
	//delete results
	delete[] results;

	/*
	//resource usage stats
    struct rusage ru;
    rv = getrusage(RUSAGE_SELF, &ru); assert(rv == 0);
    auto cv = [](const timeval &tv) {
        return double(tv.tv_sec) + double(tv.tv_usec)/1000000;
    };
    std::cerr << "Resource Usage:\n";
    std::cerr << "    User CPU Time: " << cv(ru.ru_utime) << '\n';
    std::cerr << "    Sys CPU Time: " << cv(ru.ru_stime) << '\n'; 
    std::cerr << "    Max Resident: " << ru.ru_maxrss << '\n';
    std::cerr << "    Page Faults: " << ru.ru_majflt << '\n';
	*/
	return 0;
}
