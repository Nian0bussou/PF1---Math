using System.Runtime.InteropServices;

namespace PF1;

// regroups every method related to Matrix<T> 
public struct M : IFuncs
{
    public static Matrix<T> MatNulle<T>(int rows, int cols)
    {
        var r = new List<List<T>>();
        for (int i = 0; i < rows; i++)
            r.Add(Enumerable
                  .Repeat(default(T), cols)
                  .ToList()
                  );
        return new Matrix<T>(r);
    }

    public static Matrix<double> MatRand(int rows, int cols)
    {
        var r = new Random();
        var E = MatNulle<double>(rows, cols);

        for (int i = 0; i < E.mat.Count; i++)
            E.mat[i][i] = r.Next(0, 30);

        return E;
    }

    public static Matrix<double> MatId(int x)
    {
        Matrix<double> I = MatNulle<double>(x, x);

        for (int i = 0; i < x; i++)
            I.mat[i][i] = 1;

        return I;
    }

    public static Matrix<double> MatT(Matrix<double> A)
    {
        int nRow = A.mat.Count;
        int nCol = A.mat[0].Count;
        Matrix<double> T = MatNulle<double>(nCol, nRow);

        for (int i = 0; i < nRow; i++)
            for (int j = 0; j < nRow; j++)
                T.mat[j][i] = A.mat[i][j];

        return T;
    }

    public static Matrix<double> MatSomme(Matrix<double> A, Matrix<double> B)
    {
        if (A.mat.Count != B.mat.Count) throw new ArgumentException("not the same length");

        Matrix<double> E = MatNulle<double>(A.mat.Count, A.mat[0].Count);
        for (int i = 0; i < A.mat.Count; i++)
            for (int j = 0; j < A.mat[0].Count; j++)
                E.mat[i][j] = A.mat[i][j] + B.mat[i][j];
        return E;
    }

    // for (int i = 0; i < A.mat.Count; i++)
    //     for (int j = 0; j < A.mat[0].Count; j++)
    //         A.mat[i][j] *= k;
    public static Matrix<double> MatMultk(Matrix<double> A, double k)
    => new(A.mat.Select(x => x.Select(y => y * k)
                              .ToList())
                .ToList());

    public static Matrix<double> MatProduit(Matrix<double> A, Matrix<double> B)
    {
        int rowAlen = A.mat.Count;
        int colAlen = A.mat[0].Count;
        int rowBlen = B.mat.Count;
        int colBlen = B.mat[0].Count;


        if (colAlen != rowBlen)
            throw new ArgumentException("Matrixes can't be multiplied!!");

        double tmp;

        Matrix<double> E = MatNulle<double>(rowAlen, colBlen);

        for (int i = 0; i < rowAlen; i++)
            for (int j = 0; j < colBlen; j++)
            {
                tmp = 0;
                for (int k = 0; k < colAlen; k++)
                    tmp += A.mat[i][k] * B.mat[k][j];
                E.mat[i][j] = tmp;
            }

        return E;
    }

    public static Matrix<double> MatPow(Matrix<double> A, double k)
    => k == 0 ? throw new Exception("Hoi cunt dis ai3nt define moight")
            : k == 1 ? A : MatProduit(A, MatPow(A, k - 1));

    public static double MatDetW(Matrix<double> M)
    {
        int n = M.mat.Count;
        if (n == 1) return M.mat[0][0];
        if (n == 2) return M.mat[0][0] * M.mat[1][1] - M.mat[0][1] * M.mat[1][0];

        double determinant = 0;

        for (int j = 0; j < n; j++)
            determinant +=
                ((j % 2 == 0) ? 1 : -1) // sign
                * M.mat[0][j]
                * MatDetW(MatSousMat(M, 0, j));

        return determinant;
    }

    /// <summary>
    /// Function to get the minor of a matrix by removing the specified row and column
    /// Does not check the value rowToRemove && colToRemove beforehand
    /// </summary>
    public static Matrix<double> MatSousMat(Matrix<double> matrix, int rowToRemove, int colToRemove)
    {

        List<List<double>> minor = [];

        var rows = matrix.mat.Count;
        var cols = matrix.mat[0].Count;

        int idx = 0;
        for (int i = 0; i < rows; i++)
            if (i != rowToRemove)
            {
                List<double> newRow = [];
                int jdx = 0;
                for (int j = 0; j < cols; j++)
                    if (j != colToRemove)
                    {
                        newRow.Add(matrix.mat[i][j]);
                        jdx++;
                    }
                minor.Add(newRow);
                idx++;
            }

        return new Matrix<double>(minor);
    }

    public static Matrix<double> MatInvW(Matrix<double> A)
    {

        var mA = A.mat.Count;
        var nA = A.mat[0].Count;


        if (mA != nA)
            throw new ArgumentException("MATINTW: ERR; (not a square matrix);\nAy yoo cunt m8t Nicht definiert");
        if (MatDetW(A) == 0)
            throw new ArgumentException("N'est pas inversible");

        Matrix<double> C = MatNulle<double>(mA, nA);
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                C.mat[i][j] = Math.Pow(-1, i + j) * MatDetW(MatSousMat(A, i, j));

        return MatMultk(MatT(C), 1f / MatDetW(A));
    }


    //////////////////////////////////////////////////////////////////////////////////////////// PF1 

    //given code
    public static Matrix<double> MatProduitListe(List<Matrix<double>> L)
    {
        int n = L.Count;
        int mA = L[0].mat.Count;
        int nA = L[0].mat[0].Count;
        Matrix<double> A;
        A = L[0];
        for (int i = 0; i < n; i++)
            A = MatProduit(A, L[i]);
        return A;
    }

    //given code
    public static bool MatEqual(Matrix<double> A, Matrix<double> B)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;
        int mB = B.mat.Count;
        int nB = B.mat[0].Count;
        if ((mA != mB) || (nA != nB))
            return false;
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                if (A.mat[i][j] != B.mat[i][j]) return false;
        return true;
    }
    public static Matrix<double> MatAugment(Matrix<double> A, Matrix<double> B)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;
        int mB = B.mat.Count;
        int nB = B.mat[0].Count;

        if ((mA != mB) || (nA != nB))
            throw new ArgumentException("ERR: MatAugment;; doivent avoir meme nombre de lignes");

        Matrix<double> E = MatNulle<double>(mA, nA + nB);
        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < nA; j++)
                E.mat[i][j] = A.mat[i][j];
            for (int j = 0; j < nB; j++)
                E.mat[i][nA + j] = B.mat[i][j];
        }
        return E;
    }
    public static Matrix<double> MatBackSub(Matrix<double> A, Matrix<double> B)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;
        Matrix<double> M = MatNulle<double>(nA, 1);
        double S;
        M.mat[nA - 1][0] = B.mat[nA - 1][0] / A.mat[mA - 1][nA - 1];
        for (int i = 1; i < mA; i++)
        {
            S = 0;
            for (int k = mA - i; k < nA; k++) S += A.mat[mA - i - 1][k] * M.mat[k][0];
            M.mat[mA - i - 1][0] = 1 / A.mat[mA - i - 1][nA - i - 1] * (B.mat[mA - i - 1][0] - S);
        }
        return M;
    }
    public static Matrix<double> MatForwardSub(Matrix<double> A, Matrix<double> B)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;
        Matrix<double> M = MatNulle<double>(nA, 1);
        double S;
        M.mat[0][0] = B.mat[0][0] / A.mat[0][0];
        for (int i = 1; i < mA; i++)
        {
            S = 0;
            for (int k = 0; k < i; k++)
                S += A.mat[i][k] * M.mat[k][0];
            M.mat[i][0] = 1 / A.mat[i][i] * (B.mat[i][0] - S);
        }
        return M;
    }
    public static Matrix<double> MatColumn(Matrix<double> A, int j)
    {
        int mA = A.mat.Count;
        Matrix<double> M = MatNulle<double>(mA, 1);
        for (int k = 0; k < mA; k++)
            M.mat[k][0] = A.mat[k][j - 1];
        return M;
    }
    public static Matrix<double> MatRow(Matrix<double> A, int i)
    {
        int nA = A.mat[0].Count;
        Matrix<double> M = MatNulle<double>(1, nA);
        for (int k = 0; k < nA; k++)
            M.mat[0][k] = A.mat[i - 1][k];
        return M;
    }

    public static List<Matrix<double>> ReverseL(List<Matrix<double>> L)
    {
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

    public static Matrix<double> MatEk(Matrix<double> A, int k) // Produit la matrice élémentaire pour l'échelonnage selon Gauss
    {
        int mA = A.mat.Count;

        Matrix<double> M = MatNulle<double>(mA, mA);

        if (A.mat[k - 1][k - 1] == 0)
            return MatId(mA);

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < mA; j++)
            {
                if (i == j)
                    M.mat[i][j] = 1;
                if (j == k - 1)
                    for (int s = k; s < mA; s++)
                        M.mat[s][j] = -A.mat[s][j] / A.mat[k - 1][j];
            }
        return M;
    }

    public static Matrix<double> MatEkInv(Matrix<double> A) // Produit la matrice élémentaire inverse de l'échelonnage selon Gauss
    {
        int mA = A.mat.Count;
        Matrix<double> M = MatNulle<double>(mA, mA);
        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < mA; j++)
            {
                if (i == j)
                    M.mat[i][j] = 1;
                else if (A.mat[i][j] != 0)
                    M.mat[i][j] = -A.mat[i][j];
            }
        }
        return M;
    }

    public static Matrix<double> MatPkl(int n, int k, int l) // Produit la matrice de permutation des lignes k et l
    {
        Matrix<double> M = MatNulle<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == j) & ((i != k - 1) & (i != l - 1)))
                    M.mat[i][j] = 1;
                else if (i == k - 1)
                {
                    M.mat[i][l - 1] = 1;
                    M.mat[i][k - 1] = 0;
                }
                else if (i == l - 1)
                {
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

    public static Matrix<double> MatMkl(int n, double k, int l) // Produit la matrice de multiplication d'une ligne
    {
        Matrix<double> M = MatNulle<double>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == j) & (i != l - 1))
                    M.mat[i][j] = 1;
                else if (i == l - 1)
                {
                    M.mat[i][l - 1] = k;
                }
            }
        }
        return M;
    }


    // PLU

    public static Matrix<double> MatLUlower(Matrix<double> A)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;

        Matrix<double> M = MatNulle<double>(mA, mA);
        List<Matrix<double>> E = [];

        if (mA != nA)
            throw new Exception("MatLUlower : ERREUR; La matrice doit être carrée.");

        M = A;

        for (int i = 1; i < nA; i++)
        {
            E = E.Prepend(MatEk(M, i)).ToList();
            M = MatProduit(E[0], M);
        }

        E = ReverseL(E);

        List<Matrix<double>> R = [];

        for (int i = 0; i < E.Count; i++)
            R.Add(MatEkInv(E[i]));

        return MatProduitListe(R);
    }

    public static double MatSignature(Matrix<double> P) // On assume que P est bel et
    {
        int n = P.mat.Count;
        // On crée une matrice 2xn représentant les permutations
        int[,] M = new int[2, n];
        for (int i = 0; i < n; i++)
        {
            M[0, i] = i + 1;
            for (int k = 0; k < n; k++)
            {
                if (P.mat[i][k] == 1)
                {
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

    public static int MatCherchePivot(Matrix<double> A, int h, int k)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;

        double m = Math.Abs(A.mat[h][k - 1]);

        int LignePivot = h;

        for (int i = h + 1; i < mA; i++)
        {
            if (Math.Abs(A.mat[i][k - 1]) > m)
            {
                m = Math.Abs(A.mat[i][k - 1]);
                LignePivot = i;
            }
        }

        return m == 0 ? 0 : LignePivot + 1;
    }

    public static Matrix<double> MatEkGJ(Matrix<double> A, int h, int k)
    {
        int mA = A.mat.Count;
        Matrix<double> M = MatId(mA);

        for (int i = 0; i < mA; i++)
            if (i != h - 1)
                M.mat[i][h - 1] = -A.mat[i][k - 1] / A.mat[h - 1][k - 1];
        return M;
    }

    public static int[,] MatPivots(Matrix<double> A)
    {
        var mA = A.mat.Count;
        var nA = A.mat[0].Count;
        var B = MatRREF(A);

        List<int[]> Lpivots = [];

        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nA; j++)
                //if (B[i,j]!=0) // provoque une instabilité numérique
                if (Math.Abs(B.mat[i][j]) > Math.Pow(10, -12))
                {
                    Lpivots.Add([i + 1, j + 1]);
                    break;
                }

        int[,] M = new int[Lpivots.Count, 2];

        for (int i = 0; i < Lpivots.Count; i++)
        {
            M[i, 0] = Lpivots[i][0];
            M[i, 1] = Lpivots[i][1];
        }
        return M;
    }

    public static void MatAfficheString<T>(Matrix<T> A)
    {
        for (int i = 0; i < A.mat.Count; i++)
        {
            Console.WriteLine();
            for (int j = 0; j < A.mat[i].Count; j++)
                Console.Write(string.Format("{0}", A.mat[i][j]));
            Console.WriteLine();
        }
    }
    public static void MatSol(Matrix<double> A, Matrix<double> B)
    {
        int mA = A.mat.Count;
        int nA = A.mat[0].Count;
        int mB = B.mat.Count;
        if (mA != mB)
        {
            Matrix<string> ES = MatNulle<string>(1, 1);
            ES.mat[0][0] = "ERREUR: Les dimensions sont incompatibles.";
            MatAfficheString(ES);
        }
        else if (MatRank(A) < MatRank(MatAugment(A, B)))
        {
            Matrix<string> ES = MatNulle<string>(1, 1);
            ES.mat[0][0] = "Aucune solution.";
            MatAfficheString(ES);
        }
        else
        {
            Matrix<string> ES = MatNulle<string>(1, 1);
            Matrix<double> X = MatSolve(A, B);

            for (int i = 0; i < nA; i++)
            {
                ES.mat[i][0] = "x_" + (i + 1) + " = ";
                ES.mat[i][1] = X.mat[i][0].ToString("0.0000");

                for (int j = 1, k = 1; j < X.mat[0].Count; j++)
                {
                    if (Math.Abs(X.mat[i][j]) < Math.Pow(10, -12)) // On skippe si on est trop près de "0".
                    {
                        ES.mat[i][2 * j] = " ";
                        ES.mat[i][2 * j + 1] = " ";
                    }
                    else
                    {
                        if (X.mat[i][j] > 0)
                        {
                            ES.mat[i][2 * j] = " + ";
                            ES.mat[i][2 * j + 1] = Math.Abs(X.mat[i][j]).ToString() + "t_" + (k);
                        }
                        else
                        {
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
    //////////////////////////////////////////////////////////////////////////////////////////// 
    public static Matrix<double> ImportMatrix(string chemin)
    {
        string firstLine = File.ReadLines(chemin).First();
        int ColumnsCount = 0;

        ColumnsCount =
            firstLine
            .Split(
                '\t',
                StringSplitOptions.RemoveEmptyEntries)
            .Length;

        return new Matrix<double>(File
            .ReadAllText(chemin)
            .Split(
                Array.Empty<string>(),
                StringSplitOptions.RemoveEmptyEntries)
            .Select(
                (s, i) => new
                {
                    N = double.Parse(s),
                    I = i
                })
            .GroupBy(
                at => at.I / ColumnsCount,
                at => at.N,
                (k, g) => g.ToList())
            .ToList());
    }

    //////////////////////////////////////////////////////////////////////////////////////////// Code to do


    public static Matrix<double> MatTestP(Matrix<double> A, int k)
    {
        var P = MatId(A.mat.Count);

        int idx = k;

        // find a row where the value at k isn't '0'
        double value = Math.Abs(A.mat[k][k]);
        for (int i = k + 1; i < A.mat.Count; i++)
            if (Math.Abs(A.mat[i][k]) != 0)
                idx = i;
        if (idx != k)
            for (int i = 0; i < A.mat.Count; i++)
                (P.mat[idx][i], P.mat[k][i])
                = (P.mat[k][i], P.mat[idx][i]);

        int x = 2; // avoid unreachable code warning
        if (x == 2)
            throw new NotImplementedException();

        return P;
    }


    // TODO: implement these functions:
    public static
        (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
        MatLu(Matrix<double> A)
    {
        throw new NotImplementedException();
    }

    public static
        Matrix<double> MatSolve(Matrix<double> A, Matrix<double> B)
    {
        throw new NotImplementedException();
    }

    public static double MatDet(Matrix<double> A)
    {
        throw new NotImplementedException();
    }

    public static Matrix<double> MatInvLU(Matrix<double> A)
    {
        throw new NotImplementedException();
    }

    public static Matrix<double> MatRREF(Matrix<double> A)
    {
        throw new NotImplementedException();
    }

    public static double MatRank(Matrix<double> A)
    {
        throw new NotImplementedException();
    }
}