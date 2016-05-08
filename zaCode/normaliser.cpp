#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

const int MAX_LINE_NUM = 256;

/// read a single column for csv file
std::vector<double>
read_data(char **argv, int col_num, FILE *f);

/// normalise vector by using x = (x - mean) / stdev;
void
normalise(std::vector<double>& v);

/// implementation of reading lines that does not overflow
inline void
read_ln(char *p, FILE *f, int max_len);

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "usage " << *argv
	      << " <csv-file> <col-name-to-norm>" << std::endl;
    exit(EXIT_FAILURE);
  }

  // use c file i/o, which is faster than streams
  FILE *f = fopen(argv[1], "r");

  char line[MAX_LINE_NUM];
  // first line has headers
  read_ln(line, f, MAX_LINE_NUM);

  int cols = 0;
  char *p = line;
  while (*p) {
    if (*p++ == ',') {
      cols++;
    }
  }

  int col_num = 0;
  p = line;
  char *end = strstr(line, argv[2]);
  while (p != end) {
    if (*p++ == ',') {
      col_num++;
    }
  }

  std::cout << "asked to normalise col " << argv[2] << " ("
	    << col_num << " out of " << cols << ")" << std::endl;
  
  auto v = read_data(argv, col_num, f);

  normalise(v);
  std::string out_colname = "normalised-" + std::string(argv[2]);
  std::string out_path = out_colname + ".csv"; 

  FILE *out = fopen(out_path.c_str(), "w");

  fprintf(out, "%s\n", out_colname.c_str());
  for (double x : v) {
    fprintf(out, "%lf\n", x);
  }
  fclose(out);

  std::cout << "wrote output to " << out_path << std::endl;
  
  return 0;
}

void
normalise(std::vector<double>& v)
{
  double sz = v.size();
  double mean = std::accumulate(v.begin(), v.end(), 0) / sz;
  
  double stdev = std::accumulate(v.begin(), v.end(), 0,
				 [mean](double sum, double d)
				 {
				   return sum + (d - mean) * (d - mean); 
				 });
  stdev = std::sqrt(stdev / sz);

  std::cout << "mean " << mean << std::endl << "stdev " << stdev << std::endl;

  for (auto iter = v.begin(); iter != v.end(); ++iter) {
    *iter = (*iter - mean) / stdev;
  }

}

inline void
read_ln(char *p, FILE *f, int max_len)
{
  p[max_len-1] = 'A'; // put overflow marker

  char fmt[50];
  sprintf(fmt, "%%%ds", max_len-1);

  fscanf(f, fmt, p);
  if (p[max_len-1] != 'A') {
    std::cerr << "overflow on reading line\n"
                 "please change MAX_LINE_NUM and recompile" << std::endl;
    exit(EXIT_FAILURE);
  }
}

std::vector<double>
read_data(char **argv, int col_num, FILE *f)
{
  std::vector<double> v;
  v.reserve(2400000);
 
  auto iter = v.begin();
  char line[MAX_LINE_NUM];
  char fmt[50];
  sprintf(fmt, "%%%ds", MAX_LINE_NUM-1);
  
  while(!feof(f)) {
    // read line, skip cols not needed
    line[MAX_LINE_NUM-1] = 'A';
    fscanf(f, fmt, line);
    if (line[MAX_LINE_NUM-1] != 'A') {
      std::cerr << "overflow on reading line\n"
	"please change MAX_LINE_NUM and recompile" << std::endl;
      exit(EXIT_FAILURE);
    } 

    char *p = line-1;
    int cnt_col = 0;
    while (cnt_col < col_num) {
      if (*++p == ',') cnt_col++;
    }

    char *begin = p + 1;

    while (*++p != ',');
    p[-1] = '\0'; // to stop atof
    
    v.push_back(atof(begin));
  }

  std::cout << "read in " << v.size() << " values" << std::endl;

  fclose(f);
  
  return v;
}
