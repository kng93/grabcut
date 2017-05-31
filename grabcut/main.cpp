#include "common.h"

#ifdef OCV
#include "opencv_grabcut.h"
#elif defined(ARM)
#include "arma_grabcut.h"
#endif

int main(int argc, char** argv)
{
	// Image name
	int ret_val = -1;
	#ifdef OCV
		ret_val = run_opencv();
	#elif defined(ARM)
		ret_val = run_arma();
	#endif
	return ret_val;
}
