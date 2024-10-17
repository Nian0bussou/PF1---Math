#pragma warning disable CS8321 // Local function is declared but never used

using PF1;
using MF = PF1.MatFuncs;
using SF = PF1.SplinesFunc;

static void WRT<T>(T t, bool newline = true) { if (newline) Console.WriteLine(t); else Console.Write($"{t}, "); }
static void WRTARR<T>(T[] a, bool newline = true) => a.ToList().ForEach(t => { WRT(t, false); if (newline) Console.WriteLine(); });
static void WRTLST<T>(List<T> a, bool newline = true) => a.ToList().ForEach(t => { WRT(t, false); if (newline) Console.WriteLine(); });



var prob1 = () => {
    WRT("Probleme 1"); var A = new qmatrix<double>([
           [  1,  2,  1],
    [ -1, -4,  5],
    [  2,  0, 21],
    ]);
    var l = MF.MatLu(A);
    WRT("P:");
    WRT(l[0]);
    WRT("L:");
    WRT(l[1]);
    WRT("U:");
    WRT(l[2]);
    WRT("Det A");
    WRT(MF.MatDet(A));
    WRT("Inv A");
    WRT(l[3]);
};

var prob2 = () => {
    WRT("Probleme 2");
    var A = new qmatrix<double>([
        [  1,  2, -3,  1],
        [  2,  4, -5,  3],
        [ -3, -6, 11, -2],
        [  5, 13,-15,  8],
    ]);
    var l = MF.MatLu(A);
    WRT("P:");
    WRT(l[0]);
    WRT("L:");
    WRT(l[1]);
    WRT("U:");
    WRT(l[2]);
    WRT("Det A");
    WRT(MF.MatDet(A));
    WRT("Inv A");
    WRT(l[3]);
};

var prob3 = () => {
    WRT("Probleme 3");
    var A = new qmatrix<double>([
        [  3, -1,  5],
        [  6, -3, -4],
        [  0,  1,  3],
    ]);
    var B = new qmatrix<double>([
        [ -4],
        [ -7],
        [  0],
    ]);
    // var PLUI = MF.MatLu(A);
    // var P = PLUI[0];
    // var L = PLUI[1];
    // var U = PLUI[2];
    // var I = PLUI[3];

    // var Y = MF.ForwardSubstitution(L, MF.MatProduit(I, B));
    // var X = MF.BackwardSubstitution(U, Y);
    WRT(MF.MatSolve(A, B));
};

var prob4 = () => {
    WRT("Probleme 4");
    // {
    //     // (a)
    //     var A = new qmatrix<double>([
    //             [ 1,-3, 4, 1],
    //             [ 1, 1,-1, 0],
    //         ]);
    //     var B = new qmatrix<double>([
    //             [ 12],
    //             [ -8],
    //         ]);
    //     var Noyau = MF.MatKer(A);
    //     var LGL = MF.MatRREF(A);
    //     var GJ = LGL[0];
    //     var PI = LGL[1];
    //     var solution = MF.MatSolve(A, B);
    //     WRT(GJ);
    //     WRT(PI);
    //     WRT(solution);
    // };
    {
        // (b)
        var A = new qmatrix<double>([
                [ 0, 1, 0, 0, 1],
                [ 0, 0, 0, 1, 1],
                [ 0, 0, 1, 1, 1],
                [ 1,-2, 1,-1, 0],
            ]);
        var B = new qmatrix<double>([
                [ 2],
                [-2],
                [ 3],
                [ 0],
            ]);
        var Noyau = MF.MatKer(A);
        var LGL = MF.MatRREF(A);
        var GJ = LGL[0];
        var PI = LGL[1];
        WRT("GJ: ");
        WRT(GJ);
        WRT("PI: ");
        WRT(PI);

        WRT("Sol; ");
        WRT(MF.MatSolve(A, B));
        // WRT(solution);
    }
};

var prob5 = () => {
    List<(int, double)> Lp = [(-2, 3), (-4, 5), (-1, 3), (0, 0), (3, 2), (4, -3), (2, -1), (0, -2), (-3, 0), (-2, 3)];

    var Lx = Lp.Select(x => x.Item1).ToList();
    var Ly = Lp.Select(x => x.Item2).ToList();

    var asdf = SF.SplineF(Lx, Ly);

    WRT(asdf);
};

// prob1();
// prob2();
// prob3();
// prob4();
prob5();
