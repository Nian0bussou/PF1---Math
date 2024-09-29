#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.

namespace PF1;

// regroups every method related to Matrix<T> 
public struct M : IFuncs {
    /// <summary>
    /// sometime will output somevalues that are '-0'
    /// || 0 -> 0
    /// (probably caused by floating point precision error)
    /// </summary>
    public static qmatrix<double> MatFilterNegZero(qmatrix<double> A) {
        for (int r = 0; r < A.Rows; r++)
            for (int c = 0; c < A.Cols; c++)
                if (A.mat[r][c] == -0)
                    A.mat[r][c] = Math.Abs(A.mat[r][c]);
        return A;
    }

    /// <summary>
    /// gives a null matrix
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="rows"></param>
    /// <param name="cols"></param>
    /// <returns></returns>
    public static qmatrix<T> MatNulle<T>(int rows, int cols) {
        T[][] A = new T[rows][];
        for (int r = 0; r < rows; r++) A[r] = new T[cols];

        T[][] E = A.ToList().Select(x => x.ToList().Select(_ => default(T)).ToArray()).ToArray();

        return new qmatrix<T>(E);
    }

    /// <summary>
    /// generate a random matrix
    /// </summary>
    /// <param name="rows"></param>
    /// <param name="cols"></param>
    /// <returns></returns>
    public static qmatrix<double> MatRand(int rows, int cols) {
        var r = new Random();
        var E = MatNulle<double>(rows, cols);

        for (int i = 0; i < E.Rows; i++)
            E.mat[i][i] = r.Next(0, 10000);

        return E;
    }

    /// <summary>
    /// identity matrix
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static qmatrix<double> MatId(int x) {
        qmatrix<double> I = MatNulle<double>(x, x);

        for (int i = 0; i < x; i++)
            I.mat[i][i] = 1;

        return I;
    }

    /// <summary>
    /// transpose the matrix
    /// </summary>
    /// <param name="A"></param>
    /// <returns></returns>
    public static qmatrix<double> MatT(qmatrix<double> A) {
        int nRow = A.Rows;
        int nCol = A.Cols;
        qmatrix<double> T = MatNulle<double>(nCol, nRow);

        for (int i = 0; i < nRow; i++)
            for (int j = 0; j < nRow; j++)
                T.mat[j][i] = A.mat[i][j];

        return T;
    }

    /// <summary>
    /// A + B
    /// </summary>
    /// <exception cref="ArgumentException">returns an exception if A & B are not the same size</exception>
    public static qmatrix<double> MatSomme(qmatrix<double> A, qmatrix<double> B) {
        if (A.Rows != B.Rows || A.Cols != B.Cols) throw new ArgumentException("not the same length");

        qmatrix<double> E = MatNulle<double>(A.Rows, A.Cols);
        for (int i = 0; i < A.Rows; i++)
            for (int j = 0; j < A.Cols; j++)
                E.mat[i][j] = A.mat[i][j] + B.mat[i][j];
        return E;
    }

    /// <summary>
    /// multiplies A by k
    /// </summary>
    /// <param name="A"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    public static qmatrix<double> MatMultk(qmatrix<double> A, double k)
    => new(A.mat.Select(x => x.Select(y => y * k)
                              .ToArray())
                .ToArray());

    /// <summary>
    /// Dot product of A, B
    /// </summary>
    /// <param name="A"></param>
    /// <param name="B"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static qmatrix<double> MatProduit(qmatrix<double> A, qmatrix<double> B) {
        int rowAlen = A.Rows;
        int colAlen = A.Cols;
        int rowBlen = B.Rows;
        int colBlen = B.Cols;


        if (colAlen != rowBlen) throw new ArgumentException("Matrixes can't be multiplied!!");

        double tmp;

        qmatrix<double> E = MatNulle<double>(rowAlen, colBlen);

        for (int i = 0; i < rowAlen; i++) {
            for (int j = 0; j < colBlen; j++) {
                tmp = 0;
                for (int k = 0; k < colAlen; k++)
                    tmp += A.mat[i][k] * B.mat[k][j];
                E.mat[i][j] = tmp;
            }
        }

        return E;
    }

    /// <summary>
    /// power of a matrix A by k,  <br>
    /// uses recursion
    /// </summary>
    /// <exception cref="Exception">matrix not defined with k == 0 </exception>
    public static qmatrix<double> MatPow(qmatrix<double> A, double k)
    => k == 0 ? throw new Exception("Hoi cunt dis ai3nt define moight")
            : k == 1 ? A : MatProduit(A, MatPow(A, k - 1));

    /// <summary>
    /// determinant of the matrix
    /// </summary>
    public static double MatDetW(qmatrix<double> M) {
        int n = M.Rows;
        if (n == 1) return M.mat[0][0];
        if (n == 2) return M.mat[0][0] * M.mat[1][1] - M.mat[0][1] * M.mat[1][0];

        double determinant = 0;

        for (int j = 0; j < n; j++)
            determinant +=
                ((j % 2 == 0) ? 1 : -1) // signe du determinant
                * M.mat[0][j]
                * MatDetW(MatSousMat(M, 0, j));

        return determinant;
    }

    /// <summary>
    /// Function to get the minor of a matrix by removing the specified row and column
    /// Does not check the value rowToRemove && colToRemove beforehand
    /// </summary>
    public static qmatrix<double> MatSousMat(qmatrix<double> matrix, int rowToRemove, int colToRemove) {

        double[][] minor = [];

        var rows = matrix.Rows;
        var cols = matrix.Cols;

        var iminor = 0;
        for (int i = 0; i < rows; i++) {
            var jminor = 0;
            if (i != rowToRemove) // skip the row to remove
            {
                for (int j = 0; j < cols; j++)
                    if (j != colToRemove) // skip the column to remove
                    {
                        minor[iminor][jminor] = matrix.mat[i][j];
                        jminor += 1;
                    }
                iminor += 1;
            }
        }

        return new qmatrix<double>(minor);
    }

    public static qmatrix<double> MatInvW(qmatrix<double> A) {

        var mA = A.Rows;
        var nA = A.Cols;


        if (mA != nA) throw new ArgumentException("MATINTW: ERR; (not a square matrix);\nAy yoo cunt m8t Nicht definiert");
        if (MatDetW(A) == 0) throw new ArgumentException("N'est pas inversible");

        qmatrix<double> C = MatNulle<double>(mA, nA);

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                C.mat[i][j] = Math.Pow(-1, i + j) * MatDetW(MatSousMat(A, i, j));

        return MatMultk(MatT(C), 1f / MatDetW(A));
    }


    //////////////////////////////////////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////////////////////////////////////// 


    /////////////////////////////////////// 
    // THIS SECTION CONTAINS CODE GIVEN. // 
    /////////////////////////////////////// 


    public static qmatrix<double> MatProduitListe(List<qmatrix<double>> L) {
        int n = L.Count;
        int mA = L[0].Rows;
        int nA = L[0].Cols;
        qmatrix<double> A;
        A = L[0];
        for (int i = 0; i < n; i++) A = MatProduit(A, L[i]);
        return A;
    }

    public static bool MatEqual(qmatrix<double> A, qmatrix<double> B) {
        int mA = A.Rows;
        int nA = A.Cols;
        int mB = B.Rows;
        int nB = B.Cols;
        if ((mA != mB) || (nA != nB))
            return false;
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                if (A.mat[i][j] != B.mat[i][j]) return false;
        return true;
    }
    public static qmatrix<double> MatAugment(qmatrix<double> A, qmatrix<double> B) {
        int rA = A.Rows;
        int cA = A.Cols;
        int rB = B.Rows;
        int cB = B.Cols;

        if (rA != rB) throw new ArgumentException("ERR: MatAugment;; doivent avoir meme nombre de lignes");

        qmatrix<double> E = MatNulle<double>(rA, cA + cB);
        for (int i = 0; i < rA; i++) {
            for (int j = 0; j < cA; j++)
                E.mat[i][j] = A.mat[i][j];
            for (int j = 0; j < cB; j++)
                E.mat[i][cA + j] = B.mat[i][j];
        }
        return E;
    }
    public static qmatrix<double> MatBackSub(qmatrix<double> A, qmatrix<double> B) {
        int mA = A.Rows;
        int nA = A.Cols;
        qmatrix<double> M = MatNulle<double>(nA, 1);
        double S;
        M.mat[nA - 1][0] = B.mat[nA - 1][0] / A.mat[mA - 1][nA - 1];
        for (int i = 1; i < mA; i++) {
            S = 0;
            for (int k = mA - i; k < nA; k++)
                S += A.mat[mA - i - 1][k] * M.mat[k][0];

            M.mat[mA - i - 1][0] =
                1 / A.mat[mA - i - 1][nA - i - 1] * (B.mat[mA - i - 1][0] - S);
        }
        return M;
    }
    public static qmatrix<double> MatForwardSub(qmatrix<double> A, qmatrix<double> B) {
        int mA = A.Rows;
        int nA = A.Cols;

        qmatrix<double> M = MatNulle<double>(nA, 1);

        double S;

        M.mat[0][0] = B.mat[0][0] / A.mat[0][0];

        for (int i = 1; i < mA; i++) {
            S = 0;
            for (int k = 0; k < i; k++)
                S += A.mat[i][k] * M.mat[k][0];
            M.mat[i][0] =
                1 / A.mat[i][i] * (B.mat[i][0] - S);
        }
        return M;
    }
    public static qmatrix<double> MatColumn(qmatrix<double> A, int j) {
        int mA = A.Rows;
        qmatrix<double> M = MatNulle<double>(mA, 1);
        for (int k = 0; k < mA; k++)
            M.mat[k][0] = A.mat[k][j - 1];
        return M;
    }
    public static qmatrix<double> MatRow(qmatrix<double> A, int i) {
        int nA = A.Cols;
        qmatrix<double> M = MatNulle<double>(1, nA);
        for (int k = 0; k < nA; k++)
            M.mat[0][k] = A.mat[i - 1][k];
        return M;
    }

    public static List<qmatrix<double>> ReverseL(List<qmatrix<double>> L) {
        L.Reverse();
        return L;
    }
    // static List<double[,]> ReverseL(List<double[,]> L)
    // {
    //     int n = L.Count;
    //     List<double[,]> Lrev = new List<double[,]>();
    //     for (int i = 0; i < n; i++)
    //         Lrev.Add(L[n - i - 1]);
    //     return Lrev;
    // }

    public static qmatrix<double> MatEk(qmatrix<double> A, int k) // Produit la matrice élémentaire pour l'échelonnage selon Gauss
    {
        int mA = A.Rows;

        qmatrix<double> M = MatNulle<double>(mA, mA);

        if (A.mat[k - 1][k - 1] == 0)
            return MatId(mA);

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < mA; j++) {
                if (i == j)
                    M.mat[i][j] = 1;
                if (j == k - 1)
                    for (int s = k; s < mA; s++)
                        M.mat[s][j] = -A.mat[s][j] / A.mat[k - 1][j];
            }
        return M;
    }

    public static qmatrix<double> MatEkInv(qmatrix<double> A) // Produit la matrice élémentaire inverse de l'échelonnage selon Gauss
    {
        int mA = A.Rows;
        qmatrix<double> M = MatNulle<double>(mA, mA);
        for (int i = 0; i < mA; i++) {
            for (int j = 0; j < mA; j++) {
                if (i == j)
                    M.mat[i][j] = 1;
                else if (A.mat[i][j] != 0)
                    M.mat[i][j] = -A.mat[i][j];
            }
        }
        return M;
    }

    public static qmatrix<double> MatPkl(int n, int k, int l) // Produit la matrice de permutation des lignes k et l
    {
        qmatrix<double> M = MatNulle<double>(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == j) & ((i != k - 1) & (i != l - 1)))
                    M.mat[i][j] = 1;
                else if (i == k - 1) {
                    M.mat[i][l - 1] = 1;
                    M.mat[i][k - 1] = 0;
                }
                else if (i == l - 1) {
                    M.mat[i][k - 1] = 1;
                    M.mat[i][l - 1] = 0;
                }
            }
        }
        if (k == l)
            return MatId(n);
        else
            return M;
    }

    public static qmatrix<double> MatMkl(int n, double k, int l) // Produit la matrice de multiplication d'une ligne
    {
        qmatrix<double> M = MatNulle<double>(n, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == j) & (i != l - 1))
                    M.mat[i][j] = 1;
                else if (i == l - 1) {
                    M.mat[i][l - 1] = k;
                }
            }
        }
        return M;
    }


    // PLU

    public static qmatrix<double> MatLUlower(qmatrix<double> A) {
        int mA = A.Rows;
        int nA = A.Cols;

        qmatrix<double> M = MatNulle<double>(mA, mA);
        List<qmatrix<double>> E = [];

        if (mA != nA)
            throw new Exception("MatLUlower : ERREUR; La matrice doit être carrée.");

        M = A;

        for (int i = 1; i < nA; i++) {
            E = E.Prepend(MatEk(M, i)).ToList();
            M = MatProduit(E[0], M);
        }

        E = ReverseL(E);

        List<qmatrix<double>> R = [];

        for (int i = 0; i < E.Count; i++)
            R.Add(MatEkInv(E[i]));

        return MatProduitListe(R);
    }

    public static double MatSignature(qmatrix<double> P) // On assume que P est bel et
    {
        int n = P.Rows;
        // On crée une matrice 2xn représentant les permutations
        int[,] M = new int[2, n];
        for (int i = 0; i < n; i++) {
            M[0, i] = i + 1;
            for (int k = 0; k < n; k++) {
                if (P.mat[i][k] == 1) {
                    M[1, i] = k + 1;
                    break;
                }
            }
        }
        // On crée une liste de toutes les paires d'indices [i,j] avec j>i
        List<int[]> C = [];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j > i)
                    C.Add([i + 1, j + 1]);

        double S = 1;
        foreach (int[] c in C)
            S *= (M[1, c[1] - 1] - M[1, c[0] - 1]) / (c[1] - c[0]);

        return S;
    }

    public static int MatCherchePivot(qmatrix<double> A, int nombrePivots, int colonne) {
        int mA = A.Rows;
        int nA = A.Cols;

        double m = Math.Abs(A.mat[nombrePivots][colonne - 1]);

        int LignePivot = nombrePivots;

        for (int i = nombrePivots + 1; i < mA; i++) {
            if (Math.Abs(A.mat[i][colonne - 1]) > m) {
                m = Math.Abs(A.mat[i][colonne - 1]);
                LignePivot = i;
            }
        }

        return m == 0 ? 0 : LignePivot + 1;
    }

    public static qmatrix<double> MatEkGJ(qmatrix<double> A, int h, int k) {
        int mA = A.Rows;
        qmatrix<double> M = MatId(mA);

        for (int i = 0; i < mA; i++)
            if (i != h - 1)
                M.mat[i][h - 1] = -A.mat[i][k - 1] / A.mat[h - 1][k - 1];
        return M;
    }

    public static int[,] MatPivots(qmatrix<double> A) {
        var mA = A.Rows;
        var nA = A.Cols;
        var B = MatRREF(A);

        List<int[]> Lpivots = [];

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                //if (B[i,j]!=0) // provoque une instabilité numérique
                if (Math.Abs(B.mat[i][j]) > Math.Pow(10, -12)) {
                    Lpivots.Add([i + 1, j + 1]);
                    break;
                }

        int[,] M = new int[Lpivots.Count, 2];

        for (int i = 0; i < Lpivots.Count; i++) {
            M[i, 0] = Lpivots[i][0];
            M[i, 1] = Lpivots[i][1];
        }
        return M;
    }

    public static void MatAfficheString<T>(qmatrix<T> A) {
        for (int i = 0; i < A.Rows; i++) {
            Console.WriteLine();
            for (int j = 0; j < A.Cols; j++)
                Console.Write(string.Format("{0}", A.mat[i][j]));
            Console.WriteLine();
        }
    }
    public static void MatSol(qmatrix<double> A, qmatrix<double> B) {
        int mA = A.Rows;
        int nA = A.Cols;
        int mB = B.Rows;
        if (mA != mB) {
            qmatrix<string> ES = MatNulle<string>(1, 1);
            ES.mat[0][0] = "ERREUR: Les dimensions sont incompatibles.";
            MatAfficheString(ES);
        }
        else if (MatRank(A) < MatRank(MatAugment(A, B))) {
            qmatrix<string> ES = MatNulle<string>(1, 1);
            ES.mat[0][0] = "Aucune solution.";
            MatAfficheString(ES);
        }
        else {
            qmatrix<string> ES = MatNulle<string>(1, 1);
            qmatrix<double> X = MatSolve(A, B);

            for (int i = 0; i < nA; i++) {
                ES.mat[i][0] = "x_" + (i + 1) + " = ";
                ES.mat[i][1] = X.mat[i][0].ToString("0.0000");

                for (int j = 1, k = 1; j < X.Cols; j++) {
                    if (Math.Abs(X.mat[i][j]) < Math.Pow(10, -12)) // On skippe si on est trop près de "0".
                    {
                        ES.mat[i][2 * j] = " ";
                        ES.mat[i][2 * j + 1] = " ";
                    }
                    else {
                        if (X.mat[i][j] > 0) {
                            ES.mat[i][2 * j] = " + ";
                            ES.mat[i][2 * j + 1] = Math.Abs(X.mat[i][j]).ToString() + "t_" + (k);
                        }
                        else {
                            ES.mat[i][2 * j] = " - ";
                            ES.mat[i][2 * j + 1] = Math.Abs(X.mat[i][j]).ToString() + "t_" + (k);
                        }
                    }
                    k++;
                }
            }
            MatAfficheString(ES);
        }
    }

    public static qmatrix<double> ImportMatrix(string chemin) {
        string firstLine = File.ReadLines(chemin).First();
        int ColumnsCount = 0;

        ColumnsCount =
            firstLine
            .Split(
                '\t',
                StringSplitOptions.RemoveEmptyEntries)
            .Length;

        return new qmatrix<double>(File
            .ReadAllText(chemin)
            .Split(
                Array.Empty<string>(),
                StringSplitOptions.RemoveEmptyEntries)
            .Select(
                (s, i) => new {
                    N = double.Parse(s),
                    I = i
                })
            .GroupBy(
                at => at.I / ColumnsCount,
                at => at.N,
                (k, g) => g.ToArray())
            .ToArray());
    }
    // (END) THIS SECTION CONTAINS CODE GIVEN. // 


    //////////////////////////////////////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////////////////////////////////////// 

    ///////////////////////////////////////////////// 
    // THIS SECTION CONTAINS CODE TO WRITE MYSELF. // 
    ///////////////////////////////////////////////// 


    public static qmatrix<double> MatTestP(qmatrix<double> A, int k) // given ? need to look further into that one
    {
        var P = MatId(A.Rows);

        int idx = k;

        double value = Math.Abs(A.mat[k][k]);

        for (int i = k + 1; i < A.Rows; i++)
            if (Math.Abs(A.mat[i][k]) != 0)
                idx = i;
        if (idx != k)
            for (int i = 0; i < A.Rows; i++)
                (P.mat[idx][i], P.mat[k][i])
                = (P.mat[k][i], P.mat[idx][i]);


        return P;
    }

    /// <summary>
    /// reduced row echelon form (Gauss-Jordan method)
    /// </summary>
    public static qmatrix<double> MatRREF(qmatrix<double> A) => throw new NotImplementedException();

    /// <summary>
    /// Get the rank of a matrix
    /// </summary>
    public static double MatRank(qmatrix<double> A) {
        // local func. to check if contains only zeros( [0, 0, ..., 0] )
        static bool containsAllZero(double[] a) {
            foreach (var v in a) if (v != 0) return false;
            return true;
        }

        int n = 0;
        foreach (var rows in A.mat) {
            if (containsAllZero(rows)) continue;
            else n++;
        }
        return n;
    }

    /// <summary>
    /// decompose matrix 'A' using PLU decomposition 
    /// </summary>
    /// <exception cref="ArgumentException">Matrix A is not a square</exception>
    /// TODO:  Finish this
    public static qmatrix<double>[] MatLu(qmatrix<double> A) {
        // use matek to get elementaire matrix 
        int n = A.Rows;

        if (A.Rows != A.Cols) throw new ArgumentException("Kyaputen, we need a square");

        qmatrix<double> P = MatId(n);
        qmatrix<double> L = MatId(n);
        qmatrix<double> U = CopyMatrix(A);

        List<qmatrix<double>> Es = [];

        qmatrix<double> An = new();
        for (int c = 0; c < A.Cols; c++) {
            var E = MatEk(A, c);
            Es.Add(E);

            An = MatProduit(MatProduitListe(Es), A);
        }

        U = CopyMatrix(An);

        L = MatInvW(MatProduitListe(Es));

        // P = ID


        return [P, L, U, MatInvW(P)];
        //////


        // // List<qMatrix<double>> E = [];

        // var pivot = A.Rows;

        // var P = MatNulle<double>(pivot, pivot);
        // var L = MatNulle<double>(pivot, pivot);
        // var U = MatNulle<double>(pivot, pivot);

        // // selong LU
        // for (var i = 0; i < pivot; i++)
        // {
        //     for (var k = i; k < pivot; k++)
        //     {
        //         double sum = 0;

        //         for (var j = 0; j < i; j++)
        //         {
        //             sum += L.mat[i][j] * U.mat[j][k];
        //         }

        //         U.mat[i][k] = A.mat[i][k] - sum;
        //     }

        //     for (var k = i; k < pivot; k++)
        //     {
        //         if (i == k)
        //         {
        //             L.mat[i][i] = 1;
        //         }
        //         else
        //         {
        //             double sum = 0;

        //             for (var j = 0; j < i; j++)
        //             {
        //                 sum += L.mat[k][j] * U.mat[j][i];
        //             }

        //             L.mat[k][i] = (A.mat[k][i] - sum) / U.mat[i][i];
        //         }
        //     }
        // }

        // return [P, L, U, MatInvW(P)];

        throw new NotImplementedException();
    }

    public static qmatrix<double> MatSolve(qmatrix<double> A, qmatrix<double> B) {
        throw new NotImplementedException();
    }

    public static double MatDet(qmatrix<double> A) {
        throw new NotImplementedException();
    }

    public static qmatrix<double> MatInvLU(qmatrix<double> A) {
        throw new NotImplementedException();
    }

    /// <summary>
    /// deep copy a matrix instead of reference copy
    /// </summary>
    private static qmatrix<double> CopyMatrix(qmatrix<double> A)
    => new(A.mat.Select(x => x.Select(x => x).ToArray()).ToArray());







}
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.