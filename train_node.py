import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# 사용자가 이름 바꾼 파일들에서 불러오기
from GaAN_node import SimpleKeyboardGaAN 
from dataset_node import prepare_keyboard_dataset

def main():
    # 1. 경로 설정 (사용자 환경에 맞게)
    BASE_PATH = r"/content/drive/MyDrive/src/node/1_AULA_F87_Pro"
    
    # 2. 데이터 로드
    print("데이터를 로드 중입니다. 잠시만 기다려 주세요...")
    full_dataset, label_dict = prepare_keyboard_dataset(BASE_PATH)
    
    # 3. 데이터 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 4. 모델 및 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleKeyboardGaAN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"학습 시작! (사용 장치: {device})")

    # 5. 학습 루프
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 모델 예측
            out = model(batch.x, batch.edge_index)
            
            # 정답 노드의 결과와 실제 라벨 비교
            loss = criterion(out[batch.y], batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 10 에폭마다 결과 출력
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f}")

    print("학습이 완료되었습니다!")
    # 학습된 모델 저장 (나중에 테스트할 때 필요함)
    torch.save(model.state_dict(), "gaan_node_model.pth")
    print("모델이 'gaan_node_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()