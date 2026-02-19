import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import os
from GaAN_node import SimpleKeyboardGaAN

# train_node.py는 dataset_stroke.py에서 준비된 그래프 데이터셋을 불러와 GaAN 모델을 학습하는 코드
def main():
    SAVE_PATH = "./processed_dataset_stroke.pt"
    
    if not os.path.exists(SAVE_PATH):
        print("데이터 파일이 없습니다. dataset_node.py를 먼저 실행하세요.")
        return

    # weights_only=False로 최신 버전 파이토치 에러 방지
    data_dict = torch.load(SAVE_PATH, weights_only=False)
    full_dataset = data_dict['dataset']
    label_map = data_dict['label_map']
    
    train_size = int(0.8 * len(full_dataset))
    train_dataset, test_dataset = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleKeyboardGaAN(num_classes=len(label_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"학습 시작! 장치: {device}")

    for epoch in range(1,501):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 모델에 x, edge_index, edge_attr 전달 (노드별 분류)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "gaan_node_model.pth")
    print("모든 과정이 완료되었습니다.")

if __name__ == "__main__":
    main()
