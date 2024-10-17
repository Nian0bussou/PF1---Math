#pragma warning disable CS8981 // The type name only contains lower-cased ascii characters. Such names may become reserved for the language.
namespace PF1;

// ease the use of matrix of different types with a generic type
//
// where the Matrix is declared and used from
// starts with 'Q' because faster autocompletion when typing 
public struct qmatrix<T> {
    public T[][] mat;
    public readonly int Rows { get => mat.Length; }
    public readonly int Cols { get => mat[0].Length; }
    public readonly bool isSquare { get => Rows == Cols; }

    public qmatrix(T[][] t) {
        mat = t;
    }
    public qmatrix(T[] x) {
        mat = new T[x.Length][];
        for (int i = 0; i < x.Length; i++) {
            mat[i][0] = x[i];
        }
    }



    public override readonly string ToString() {
        string s = "";
        foreach (var row in mat) {
            foreach (var x in row) {
                var msg = $"| {x:0.00000}";
                s += string.Format(msg.PadRight(15));
            }
            s += " |" + Environment.NewLine;
        }
        return s;
    }
}


#pragma warning restore CS8981 // The type name only contains lower-cased ascii characters. Such names may become reserved for the language.