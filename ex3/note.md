# Camera Calibration
## Rotation and translation recovering from QR decomposition of Tsai method

$$
 \mathbf{M} = \mathbf{QR} = \mathbf{K[R|T]} = \mathbf{KR|KT}
 \\
 \mathbf{Q}\text{ is the unitary matrix (orthonormal matrix)}
 \mathbf{R}\text{ is the upper triangle matrix}
$$

- We want to obtain K, R and T from M-QR decomposition
- Actually we need to apply RQ decomposition to
$$
\mathbf{square_M} = \mathbf{KR} = \mathbf{RQ}
$$
- relationship between QR and RQ decomposition
  - QR decomposition is obtained by Gram-Schmidt orthogonalization of columns of A, started from the first column
  - RQ decomposition is obtained by Gram-Schmidt orthogonalization of rows of A, started from the last row
- This means that we can obtain RQ decomposition by applying QR decomposition on permuted the rows and transposed matrix

- permutation of the order of rows and transposed it to obtained matrix p_M
$$
\mathbf{(PM)^T}
$$ 

- how to obtain RQ from QR decomposition
- Since we have unitary matrix $\mathbf{Q}$ by QR decomposition
$$
\mathbf{M} = \mathbf{QR} = \mathbf{Q^{-1}Q}\mathbf{QR}\mathbf{QQ^{-1}} \\
\mathbf{Q^{-1}Q^{-1}Q}\mathbf{MQ} =  \mathbf{RQ}
$$
- Since unitary matrix has property such as $\mathbf{Q^{-1}} = \mathbf{Q}$

$$
\mathbf{M} = \mathbf{QR} = \mathbf{Q^{-1}Q}\mathbf{QR}\mathbf{QQ^{-1}} \\
\mathbf{QQQ}\mathbf{MQ} =  \mathbf{RQ}
$$
