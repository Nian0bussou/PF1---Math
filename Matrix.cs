namespace PF1;

// ease the use of matrix of different types with a generic type
//
// where the Matrix is declared and used from
public struct Matrix<T>(List<List<T>> t)
{
    public List<List<T>> mat = t;

    public override readonly string ToString()
    {
        string s = "";
        mat.ForEach(row =>
        {
            row.ForEach(c => s += string.Format($"{c,25}"));
            s += Environment.NewLine;
        });

        return s;
    }
}

// easier to look at the signature  of functions with this interface
interface IFuncs
{
    abstract static
        Matrix<double>
        MatRand
        (int rows, int cols);

    abstract static
        Matrix<T>
        MatNulle<T>
        (int rows, int cols);

    abstract static
        Matrix<double>
        MatId
        (int x);

    abstract static
        Matrix<double>
        MatT
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatSomme
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatMultk
        (Matrix<double> A, double k);

    abstract static
        Matrix<double>
        MatProduit
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatPow
        (Matrix<double> A, double k);

    abstract static
        double
        MatDetW
        (Matrix<double> M);

    abstract static
        Matrix<double>
        MatSousMat
        (Matrix<double> matrix, int rowToRemove, int colToRemove);

    abstract static
        Matrix<double>
        MatInvW
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatProduitListe
        (List<Matrix<double>> L);

    abstract static
        bool
        MatEqual
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatAugment
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatBackSub
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatForwardSub
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatColumn
        (Matrix<double> A, int j);

    abstract static
        Matrix<double>
        MatRow
        (Matrix<double> A, int i);

    abstract static
        List<Matrix<double>>
        ReverseL
        (List<Matrix<double>> L);

    abstract static
        Matrix<double>
        MatEk
        (Matrix<double> A, int k);

    abstract static
        Matrix<double>
        MatEkInv
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatPkl
        (int n, int k, int l);

    abstract static
        Matrix<double>
        MatMkl
        (int n, double k, int l);

    abstract static
        Matrix<double>
        MatLUlower
        (Matrix<double> A);

    abstract static
        double
        MatSignature
        (Matrix<double> P);

    abstract static
        int
        MatCherchePivot
        (Matrix<double> A, int h, int k);

    abstract static
        Matrix<double>
        MatEkGJ
        (Matrix<double> A, int h, int k);

    abstract static
        int[,]
        MatPivots
        (Matrix<double> A);

    abstract static
        Matrix<double>
        ImportMatrix
        (string chemin);

    abstract static
        void
        MatSol
        (Matrix<double> A, Matrix<double> B);

    abstract static
        Matrix<double>
        MatTestP
        (Matrix<double> A, int k);

    abstract static
        void
        MatAfficheString<T>
        (Matrix<T> A);

    abstract static
        (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
        MatLu
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatSolve
        (Matrix<double> A, Matrix<double> B);

    abstract static
        double
        MatDet
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatInvLU
        (Matrix<double> A);

    abstract static
        Matrix<double>
        MatRREF
        (Matrix<double> A);

    abstract static
        double
        MatRank
        (Matrix<double> A);

}