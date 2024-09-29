#pragma warning disable CS8981 // The type name only contains lower-cased ascii characters. Such names may become reserved for the language.
namespace PF1;

// ease the use of matrix of different types with a generic type
//
// where the Matrix is declared and used from
// starts with 'Q' because faster autocompletion when typing 
public struct qmatrix<T>(T[][] t) {
    public T[][] mat = t;
    public readonly int Rows { get => mat.GetLength(0); }
    public readonly int Cols { get => mat.GetLength(1); }

    public override readonly string ToString() {
        string s = "";
        mat.ToList().ForEach(row => {
            row.ToList().ForEach(c => {
                var msg = $"| {c:0.00}";
                s += string.Format(msg.PadRight(15));
            });
            s += " |" + Environment.NewLine;
        });
        return s;
    }
}

// easier to look at the signature  of functions with this interface
interface IFuncs {
    abstract static
    bool
    MatEqual
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    double
    MatDet
    (qmatrix<double> A);

    abstract static
    double
    MatDetW
    (qmatrix<double> M);

    abstract static
    double
    MatRank
    (qmatrix<double> A);

    abstract static
    double
    MatSignature
    (qmatrix<double> P);

    abstract static
    int
    MatCherchePivot
    (qmatrix<double> A, int h, int k);

    abstract static
    int[,]
    MatPivots
    (qmatrix<double> A);

    abstract static
    List<qmatrix<double>>
    ReverseL
    (List<qmatrix<double>> L);

    abstract static
    qmatrix<double>
    ImportMatrix
    (string chemin);

    abstract static
    qmatrix<double>
    MatAugment
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatBackSub
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatColumn
    (qmatrix<double> A, int j);

    abstract static
    qmatrix<double>
    MatEk
    (qmatrix<double> A, int k);

    abstract static
    qmatrix<double>
    MatEkGJ
    (qmatrix<double> A, int h, int k);

    abstract static
    qmatrix<double>
    MatEkInv
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatFilterNegZero
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatForwardSub
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatId
    (int x);

    abstract static
    qmatrix<double>
    MatInvLU
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatInvW
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatLUlower
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatMkl
    (int n, double k, int l);

    abstract static
    qmatrix<double>
    MatMultk
    (qmatrix<double> A, double k);

    abstract static
    qmatrix<double>
    MatPkl
    (int n, int k, int l);

    abstract static
    qmatrix<double>
    MatPow
    (qmatrix<double> A, double k);

    abstract static
    qmatrix<double>
    MatProduit
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatProduitListe
    (List<qmatrix<double>> L);

    abstract static
    qmatrix<double>
    MatRand
    (int rows, int cols);

    abstract static qmatrix<double>
    MatRow
    (qmatrix<double> A, int i);

    abstract static qmatrix<double>
    MatRREF
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatSolve
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatSomme
    (qmatrix<double> A, qmatrix<double> B);

    abstract static
    qmatrix<double>
    MatSousMat
    (qmatrix<double> matrix, int rowToRemove, int colToRemove);

    abstract static
    qmatrix<double>
    MatT
    (qmatrix<double> A);

    abstract static
    qmatrix<double>
    MatTestP
    (qmatrix<double> A, int k);

    abstract static
    qmatrix<double>[]
    MatLu
    (qmatrix<double> A);

    abstract static
    qmatrix<T>
    MatNulle<T>
    (int rows, int cols);

    abstract static
    void
    MatAfficheString<T>
    (qmatrix<T> A);

    abstract static
    void
    MatSol
    (qmatrix<double> A, qmatrix<double> B);
}
#pragma warning restore CS8981 // The type name only contains lower-cased ascii characters. Such names may become reserved for the language.