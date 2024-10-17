#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.

using System.Security.Cryptography;

using Assert = PF1.CAssert;

namespace PF1;

// # ChatGPT was used to generate some of the comments

// regroups every method related to Matrix<T> 
public class MatFuncs {

    // sometime will output somevalues such as '-0' 
    // returns => -0 -> 0
    // (probably caused by floating point precision error)
    public static qmatrix<double> MatFilterNegZero(qmatrix<double> A) {
        for (int r = 0; r < A.Rows; r++)
            for (int c = 0; c < A.Cols; c++)
                if (A.mat[r][c] == -0)
                    A.mat[r][c] = Math.Abs(A.mat[r][c]);
        return A;
    }

    /// gives a null matrix (default values, [int | double | float | etc. ] -> 0, string -> "", etc.. )
    public static qmatrix<T> MatNulle<T>(int rows, int cols) {
        T[][] A = new T[rows][];
        for (int r = 0; r < rows; r++) A[r] = new T[cols];
        T[][] E = A.ToList().Select(x => x.ToList().Select(_ => default(T)).ToArray()).ToArray();
        return new qmatrix<T>(E);
    }

    /// generate a random matrix
    public static qmatrix<double> MatRand(int rows, int cols) {
        var r = new Random();
        var E = MatNulle<double>(rows, cols);
        for (int i = 0; i < E.Rows; i++)
            E.mat[i][i] = r.Next(0, 10000);
        return E;
    }

    // not generic because an identity of a (string | struct{} | class | ... ) matrix wouldnt make sense
    public static qmatrix<double> MatId(int x) {
        qmatrix<double> I = MatNulle<double>(x, x);
        for (int i = 0; i < x; i++)
            I.mat[i][i] = 1;
        return I;
    }

    /// transpose the matrix
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
            determinant += ((j % 2 == 0) ? 1 : -1) // signe du determinant
                            * M.mat[0][j]
                            * MatDetW(MatSousMat(M, 0, j));

        return determinant;
    }

    /// <summary>
    /// Function to get the minor of a matrix by removing the specified row and column
    /// Does not check the value rowToRemove && colToRemove beforehand
    /// </summary>
    public static qmatrix<double> MatSousMat(qmatrix<double> matrix, int rowToRemove, int colToRemove) {


        var rows = matrix.Rows;
        var cols = matrix.Cols;

        double[][] minor = MatNulle<double>(rows, cols).mat;

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
        if (MatDet(A) == 0) throw new ArgumentException("N'est pas inversible");

        qmatrix<double> C = MatNulle<double>(mA, nA);

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                C.mat[i][j] = Math.Pow(-1, i + j) * MatDet(MatSousMat(A, i, j));

        return MatMultk(MatT(C), 1f / MatDet(A));
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

        double m = Math.Abs(A.mat[nombrePivots][colonne - 1]);

        if (m == 0) return 0;

        int LignePivot = nombrePivots;

        for (int i = nombrePivots + 1; i < mA; i++) {
            if (Math.Abs(A.mat[i][colonne - 1]) > m) {
                m = Math.Abs(A.mat[i][colonne - 1]);
                LignePivot = i;
            }
        }

        return LignePivot + 1;
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
        var ls = MatRREF(A);
        var B = ls[0];

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


    public static qmatrix<double> MatTestP(qmatrix<double> A, int k) {

        int n = A.Rows;

        var P = MatId(n);

        var pivotRow = MatCherchePivot(A, k, k);

        if (pivotRow != k) {
            for (int i = 0; i < n; i++) {
                (P.mat[k][i], P.mat[pivotRow][i]) = (P.mat[pivotRow][i], P.mat[k][i]);
            }
        }

        return P;
    }

    // これは塵を何ですか... ？？？
    public static List<qmatrix<double>> MatRREF(qmatrix<double> A) {
        var rows = A.Rows;
        var cols = A.Cols;

        qmatrix<double> E = MatId(rows);

        int lead = 0;
        for (int r = 0; r < rows; r++) {
            if (lead >= cols) {
                break;
            }
            int i = r;
            while (A.mat[i][lead] == 0) {
                i++;
                if (i == rows) {
                    i = r;
                    lead++;
                    if (lead == cols) {
                        return [A, E]; // if already in RREF form
                    }
                }
            }

            // swap rows i and r in both 
            (A.mat[i], A.mat[r]) = (A.mat[r], A.mat[i]);
            (E.mat[i], E.mat[r]) = (E.mat[r], E.mat[i]); // Swap in elementary matrix

            // make the lead 1 by dividing the row by A[r][lead] 
            double leadVal = A.mat[r][lead];
            for (int j = 0; j < cols; j++) {
                A.mat[r][j] /= leadVal;
            }
            for (int j = 0; j < rows; j++) {  // Also divide the elementary matrix row
                E.mat[r][j] /= leadVal;
            }

            // Eliminate all other rows in this column
            for (int i2 = 0; i2 < rows; i2++) {
                if (i2 != r) {
                    double leadFactor = A.mat[i2][lead];
                    for (int j = 0; j < cols; j++) {
                        A.mat[i2][j] -= leadFactor * A.mat[r][j];
                    }
                    for (int j = 0; j < rows; j++) { // Apply the same operation to E
                        E.mat[i2][j] -= leadFactor * E.mat[r][j];
                    }
                }
            }

            lead++;
        }

        return [A, E];
    }

    public static double MatRank(qmatrix<double> A) => MatPivots(A).Length;

    /// <summary>
    /// decompose matrix 'A' using PLU decomposition 
    /// </summary>
    /// <param name="A"></param>
    /// <returns>[P, L, U, MatInvLU(P)]</returns>    
    public static qmatrix<double>[] MatLu(qmatrix<double> A) {
        Assert.Assert(A.isSquare, "A should be square MatLu");

        int n = A.Rows;
        qmatrix<double> L = MatId(n); // initialize L as identity matrix
        qmatrix<double> U = CopyMatrix(A); // Iiitialize U as a copy of A
        qmatrix<double> P = MatId(n); // initialize P as identity matrix

        for (int k = 0; k < n; k++) {
            // find the pivot element (maximum in the column)
            int maxIndex = k;
            double maxValue = Math.Abs(U.mat[k][k]);
            for (int i = k + 1; i < n; i++) {
                if (Math.Abs(U.mat[i][k]) > maxValue) {
                    maxValue = Math.Abs(U.mat[i][k]);
                    maxIndex = i;
                }
            }

            // ssap rows in U to move the pivot element to the current row
            if (maxIndex != k) {
                SwapRows(U, k, maxIndex);
                SwapRows(P, k, maxIndex);
                if (k > 0)
                    SwapRows(L, k, maxIndex, k); // Only swap the columns up to the current pivot
            }

            // perform Gaussian elimination to form U and L
            for (int i = k + 1; i < n; i++) {
                double factor = U.mat[i][k] / U.mat[k][k];
                L.mat[i][k] = factor; // Fill L matrix

                // subtract the factor times the k-th row from the i-th row of U
                for (int j = k; j < n; j++) {
                    U.mat[i][j] -= factor * U.mat[k][j];
                }
            }
        }
        return [P, L, U, MatInvW(P)];
    }

    public static qmatrix<double> MatSolve(qmatrix<double> A, qmatrix<double> B) {
        // Assert.Assert(A.isSquare, $"A should be a square\nsquare?{A.isSquare}\nrows: {A.Rows}\ncols: {A.Cols}");
        Assert.Assert(A.Rows == B.Rows, "Expected : A Rows == B Rows");
        var lll = MatLu(A);
        var P = lll[0];
        var L = lll[1];
        var U = lll[2];
        var pB = MatProduit(P, B);
        var Y = ForwardSubstitution(L, pB);
        var X = BackwardSubstitution(U, Y);
        return X;
    }

    public static double MatDet(qmatrix<double> A) {
        if (A.Rows != A.Cols)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;

        var Tmp = CopyMatrix(A);

        // Gauss elimination
        double det = 1.0;
        for (int i = 0; i < n; i++) {
            // Search for maximum in this column (pivot)
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.Abs(Tmp.mat[k][i]) > Math.Abs(Tmp.mat[maxRow][i])) {
                    maxRow = k;
                }
            }

            // Swap maximum row with current row (if necessary)
            if (i != maxRow) {
                (Tmp.mat[maxRow], Tmp.mat[i]) = (Tmp.mat[i], Tmp.mat[maxRow]);
                det *= -1; // Swapping rows changes the sign of the determinant
            }

            // If the pivot element is zero, the matrix is singular (det = 0)
            if (Math.Abs(Tmp.mat[i][i]) < 1e-10) {
                return 0.0;
            }

            // For each row below the pivot row
            for (int k = i + 1; k < n; k++) {
                double factor = Tmp.mat[k][i] / Tmp.mat[i][i];
                for (int j = i; j < n; j++) {
                    Tmp.mat[k][j] -= factor * Tmp.mat[i][j];
                }
            }

            // Multiply the diagonal elements to get the determinant
            det *= Tmp.mat[i][i];
        }

        return det;
    }

    public static qmatrix<double> MatInvLU(qmatrix<double> A) {
        var LU = MatLu(A);
        var P = LU[0]; // Permutation matrix
        var L = LU[1]; // Lower triangular matrix
        var U = LU[2]; // Upper triangular matrix

        int n = A.Rows; // Assuming A is a square matrix
                        // Create identity matrix I of size n
        var I = MatId(n);

        // Initialize the inverse matrix X
        var X = MatNulle<double>(n, n);

        // Inverting the matrix using forward and backward substitution
        for (int i = 0; i < n; i++) {
            // Extract column i from the identity matrix I
            double[] e = new double[n];
            for (int j = 0; j < n; j++) {
                e[j] = I.mat[j][i];
            }

            // FIXME: e => constantes ??? 

            // Re => P^T * e
            var Tp = MatT(P);
            var Re = MatProduit(Tp, new qmatrix<double>(e));

            // solve LY = P^T * e (forward substitution)
            var y = ForwardSubstitution(L, Re);
            // vector

            // solve UX = y (backward substitution)
            var x = BackwardSubstitution(U, y);
            // vector

            // Place the result column into X
            for (int j = 0; j < n; j++) {
                X.mat[j][i] = x.mat[j][0];
            }
        }

        return X;
    }

    public static int[]? MatFreeVar(qmatrix<double> A) {
        return null;
    }
    public static int[]? MatLinkVar(qmatrix<double> A) {
        return null;
    }
    public static qmatrix<double>? MatKer(qmatrix<double> A) {
        return null;
    }

    ////////////////////// HELPER FUNCTIONS


    static qmatrix<double> CopyMatrix(qmatrix<double> A)
    => new(A.mat.Select(x => x.Select(x => x).ToArray()).ToArray());

    public static qmatrix<double> ForwardSubstitution(qmatrix<double> L, qmatrix<double> B) {
        int n = L.Rows;
        var Y = MatNulle<double>(n, B.Cols);
        for (int col = 0; col < B.Cols; col++)
            for (int i = 0; i < n; i++) {
                double sum = 0;

                Assert.Assert(L.mat[i][i] != 0, $"value ({L.mat[i][i]}) should not be 0\n\n{L}");

                for (int j = 0; j < i; j++)
                    sum += L.mat[i][j] * Y.mat[j][col];
                Y.mat[i][col] = (B.mat[i][col] - sum) / L.mat[i][i];
            }
        return Y;
    }

    public static qmatrix<double> BackwardSubstitution(qmatrix<double> U, qmatrix<double> Y) {
        int n = U.Rows;
        var X = MatNulle<double>(n, Y.Cols);
        for (int col = 0; col < Y.Cols; col++)
            for (int i = n - 1; i >= 0; i--) {
                double sum = 0;


                Assert.Assert(U.mat[i][i] != 0, $"value ({U.mat[i][i]}) should not be 0\n\n{U}");
                // if (Math.Abs(U.mat[i][i]) < 1e-10) {
                //     throw new Exception("Zero pivot encountered in ForwardSubstitution.");
                // }

                for (int j = i + 1; j < n; j++)
                    sum += U.mat[i][j] * X.mat[j][col];
                X.mat[i][col] = (Y.mat[i][col] - sum) / U.mat[i][i];
            }
        return X;
    }

    static void SwapRows(qmatrix<double> matrix, int row1, int row2, int startCol = 0) {
        int cols = matrix.Cols;
        for (int j = startCol; j < cols; j++) {
            (matrix.mat[row2][j], matrix.mat[row1][j]) = (matrix.mat[row1][j], matrix.mat[row2][j]);
        }
    }

}
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.
