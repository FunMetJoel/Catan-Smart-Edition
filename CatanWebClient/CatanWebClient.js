class CatanWebClient {
    constructor(ctx) {
        this.ctx = ctx;
        this.setupBoard();
        this.hexes = [];
    }

    setupBoard() {
        // Blue background
        this.ctx.fillStyle = "#00AAFF";
        this.ctx.fillRect(0, 0, 800, 600);

        // Show bos.png image
        var img = new Image();
        img.src = "images/bos.png";
        img.onload = () => {
            var roadSize = 10;
            var hexSize = 100;
            for (var i = 0; i < 5; i++) {
                for (var j = 0; j < 5; j++) {
                    var NewX = 0.866 * (hexSize + roadSize) * i + 0.433 * (hexSize + roadSize) * j;
                    var NewY = 0.75 * (hexSize + roadSize) * j;
                    this.ctx.drawImage(img, NewX, NewY, hexSize, hexSize);
                }
            }
        }



    }
}