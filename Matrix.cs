namespace PF1;

using System.Collections;




// ease the use of matrix of different types with a generic type
//
// where the Matrix is declared and used from
// starts with 'Q' because faster autocompletion when typing 
public struct qMatrix<T>(List<List<T>> t)
{
    public List<List<T>> mat = t;
    public readonly int Rows { get => mat.Count; }
    public readonly int Cols { get => mat[0].Count; }

    public override readonly string ToString()
    {
        string s = "";
        mat.ForEach(row =>
        {
            row.ForEach(c =>
            {
                var msg = $"| {c:0.00}";
                s += string.Format(msg.PadRight(15));
            });
            s += " |" + Environment.NewLine;
        });

        return s;
    }
}

// easier to look at the signature  of functions with this interface
interface IFuncs
{

    abstract static void MatAfficheString<T>(qMatrix<T> A);
    abstract static void MatSol(qMatrix<double> A, qMatrix<double> B);
    abstract static bool MatEqual(qMatrix<double> A, qMatrix<double> B);
    abstract static double MatDet(qMatrix<double> A);
    abstract static double MatDetW(qMatrix<double> M);
    abstract static double MatRank(qMatrix<double> A);
    abstract static double MatSignature(qMatrix<double> P);
    abstract static int MatCherchePivot(qMatrix<double> A, int h, int k);
    abstract static int[,] MatPivots(qMatrix<double> A);
    abstract static List<qMatrix<double>> ReverseL(List<qMatrix<double>> L);
    abstract static qMatrix<double> MatFilterNegZero(qMatrix<double> A);
    abstract static qMatrix<double> ImportMatrix(string chemin);
    abstract static qMatrix<double> MatAugment(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatBackSub(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatColumn(qMatrix<double> A, int j);
    abstract static qMatrix<double> MatEk(qMatrix<double> A, int k);
    abstract static qMatrix<double> MatEkGJ(qMatrix<double> A, int h, int k);
    abstract static qMatrix<double> MatEkInv(qMatrix<double> A);
    abstract static qMatrix<double> MatForwardSub(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatId(int x);
    abstract static qMatrix<double> MatInvLU(qMatrix<double> A);
    abstract static qMatrix<double> MatInvW(qMatrix<double> A);
    abstract static qMatrix<double> MatLUlower(qMatrix<double> A);
    abstract static qMatrix<double> MatMkl(int n, double k, int l);
    abstract static qMatrix<double> MatMultk(qMatrix<double> A, double k);
    abstract static qMatrix<double> MatPkl(int n, int k, int l);
    abstract static qMatrix<double> MatPow(qMatrix<double> A, double k);
    abstract static qMatrix<double> MatProduit(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatProduitListe(List<qMatrix<double>> L);
    abstract static qMatrix<double> MatRand(int rows, int cols);
    abstract static qMatrix<double> MatRow(qMatrix<double> A, int i);
    abstract static qMatrix<double> MatRREF(qMatrix<double> A);
    abstract static qMatrix<double> MatSolve(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatSomme(qMatrix<double> A, qMatrix<double> B);
    abstract static qMatrix<double> MatSousMat(qMatrix<double> matrix, int rowToRemove, int colToRemove);
    abstract static qMatrix<double> MatT(qMatrix<double> A);
    abstract static qMatrix<double> MatTestP(qMatrix<double> A, int k);
    abstract static qMatrix<double>[] MatLu(qMatrix<double> A);
    abstract static qMatrix<T> MatNulle<T>(int rows, int cols);
}