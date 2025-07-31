import os
import logging
from typing import Optional, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_gpu_resources():
    """
    Initialize GPU resources for FAISS if available
    
    Returns:
        GPU resources object or None if not available
    """
    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            # Initialize GPU resources
            logger.info("Initializing FAISS GPU resources")
            return faiss.StandardGpuResources()
        else:
            logger.warning("FAISS compiled without GPU support")
            return None
    except ImportError as e:
        logger.error(f"FAISS import error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing GPU resources: {e}")
        return None

def get_gpu_faiss_index(cpu_index, gpu_resources):
    """
    Convert a CPU FAISS index to a GPU index
    
    Args:
        cpu_index: FAISS index on CPU
        gpu_resources: GPU resources object
        
    Returns:
        FAISS index on GPU
    """
    try:
        import faiss
        if gpu_resources is not None:
            # Get index type
            index_type = type(cpu_index)
            
            if isinstance(cpu_index, faiss.IndexFlatL2):
                logger.info("Converting IndexFlatL2 to GPU")
                return faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            elif isinstance(cpu_index, faiss.IndexIVFFlat):
                logger.info("Converting IndexIVFFlat to GPU")
                return faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            else:
                logger.warning(f"Unsupported index type for GPU conversion: {index_type}")
                return cpu_index
        else:
            return cpu_index
    except Exception as e:
        logger.error(f"Error converting to GPU index: {e}")
        return cpu_index

def check_faiss_gpu():
    """
    Check if FAISS has GPU support enabled
    
    Returns:
        Tuple of (has_gpu_support, error_message)
    """
    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            try:
                # Try initializing GPU resources
                res = faiss.StandardGpuResources()
                
                # Simple test to confirm GPU functioning
                d = 64                           # dimension
                nb = 1000                        # database size
                nq = 10                          # number of queries
                
                # Make random vectors
                np.random.seed(1234)
                xb = np.random.random((nb, d)).astype('float32')
                xq = np.random.random((nq, d)).astype('float32')
                
                # Create GPU index
                cpu_index = faiss.IndexFlatL2(d)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                
                # Add vectors and search
                gpu_index.add(xb)
                D, I = gpu_index.search(xq, 5)  # Search 5 nearest neighbors
                
                return True, "FAISS GPU support confirmed and working"
            except Exception as e:
                return False, f"FAISS GPU initialization failed: {str(e)}"
        else:
            return False, "FAISS does not have GPU support enabled"
    except ImportError:
        return False, "FAISS is not installed"
    except Exception as e:
        return False, f"Error checking FAISS GPU support: {str(e)}"
