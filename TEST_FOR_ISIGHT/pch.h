// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します

#ifndef PCH_H
typedef double dbl;
typedef	enum status { success, failure, error } status;
typedef enum Mode { Exist_KerA, Exist_KerE, Exist_Both, Exist_None }ProgramMode;

class CheckMode {
private:
	ProgramMode Status;
public:
	void set_ProgramMode(int dim, int RankA,int RankE) {
		if (RankA != dim) {
			if (RankE != dim) {
				Status = Exist_Both;

			}
			else {
				Status = Exist_KerA;
			}
		}
		else if (RankE != dim) {
			Status = Exist_KerE;
		}
		else {
			Status = Exist_None;
		}
	}
	Mode getProgramMode() {
		return Status;
	}
};


// TODO: ここでプリコンパイルするヘッダーを追加します
#define PCH_H
#endif //PCH_H
