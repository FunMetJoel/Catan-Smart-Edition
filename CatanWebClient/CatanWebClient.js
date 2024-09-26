class Hex extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);
    }

    draw(ctx, objectCenter, objectSize) {
        // draw images/bos.png
        var img = new Image();
        img.src = "images/bos.png";
        img.onload = () => {
            ctx.drawImage(img, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            this.drawChildren(ctx, objectCenter, objectSize);
        }
    }
}

class Corner extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.fillStyle = this.color;
        // ctx.beginPath();
        // ctx.arc(objectCenter.x, objectCenter.y, objectSize.x / 2, 0, 2 * Math.PI);
        // ctx.fill();
        var img = new Image();
        img.src = "images/settlement_silver.svg";
        img.onload = () => {
            ctx.drawImage(img, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            this.drawChildren(ctx, objectCenter, objectSize);
        }
    }
}

class Road extends CanvasObject {
    constructor(px, py, sx, sy, rotation = 0 ) {
        super(px, py, sx, sy);
        this.rotation = rotation;
    }

    draw(ctx, objectCenter, objectSize) {
        var img = new Image();
        img.src = "images/road_red.svg";
        img.onload = () => {

            // rotate the image
            ctx.save();
            ctx.translate(objectCenter.x, objectCenter.y);
            ctx.rotate(this.rotation);
            ctx.translate(-objectCenter.x, -objectCenter.y);
            ctx.drawImage(img, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            ctx.restore();

            this.drawChildren(ctx, objectCenter, objectSize);
        }
    }
}

class CatanWebClient extends ObjectCanvas {
    constructor(ctx) {
        super(ctx);
        this.setupBoard();
        this.hexes = [];
    }

    setupBoard() {
        // Blue background
        this.ctx.fillStyle = "#00AAFF";
        this.ctx.fillRect(0, 0, 800, 800);

        this.ctx.fillStyle = "#000000";
        
        for(var x = 0; x < 100; x++) {
            for(var y = 0; y < 100; y++) {
                var screenCords = {x: x * 8, y: y * 8};
                this.ctx.fillRect(screenCords.x, screenCords.y, 1, 1);
            }
        }

        // var parentSquare = new Square(0, 0, 0.01, 1, "#00FF00");
        // this.addObject(parentSquare);

        // var colors = ["#FF0000", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#00FF00"];
        
        // for (var i = 0; i < 60; i++) {
        //     var childSquare = new Square(0, 0, 0.9, 0.9, colors[i%6]);
        //     parentSquare.addChild(childSquare);
        //     parentSquare = childSquare;
        // }

        var roadSize = 0.02;
        var hexSize = 0.175;

        for (var i = 0; i < 5; i++) {
            for (var j = 0; j < 5; j++) {
                if (
                    (j == 4 && i <= 1) ||
                    (j == 3 && i == 0) ||
                    (j == 0 && i >= 3) ||
                    (j == 1 && i == 4)
                ) {
                    continue;
                }
                var HexX = 0.866 * (hexSize + roadSize) * (i-1) - 0.433 * (hexSize + roadSize) * j;
                var HexY = 0.75 * (hexSize + roadSize) * (j-2);
                var hex = new Hex(HexX, HexY, hexSize, hexSize);
                this.addObject(hex);
                var textObject = new TextObject(0, 0, 0.866, 0.15, i + ", " + j, "Arial", "#000000");
                hex.addChild(textObject);
            }
        }

        for (var i = 0; i < 12; i++) {
            for (var j = 0; j < 6; j++) {

                if (
                    (j == 0 && i >= 7) ||
                    (j == 1 && i >= 9) ||
                    (j == 2 && i >= 11) ||
                    (j == 3 && i <= 0) ||
                    (j == 4 && i <= 2) ||
                    (j == 5 && i <= 4)
                ) {
                    continue;
                }
                
                var cornerX = 0.433 * (hexSize + roadSize) * (i) - 0.433 * (hexSize + roadSize) * j - 0.866 * (hexSize + roadSize) * 1.5;
                var cornerY = 0.75 * (hexSize + roadSize) * (j) - 0.25 * (hexSize + roadSize) * (i%2) - 0.75 * (hexSize + roadSize) * 2.34;
                var corner = new Corner(cornerX, cornerY, 3*roadSize, 3*roadSize);
                this.addObject(corner);
                // var dot = new Square(0, 0, 0.1, 0.1, "#FF0000");
                // corner.addChild(dot);
            }
        }

        for (var i = 0; i < 11; i++) {
            for (var j = 0; j < 11; j++) {

                if (
                    (j%2 == 1 && i % 2 == 1) ||
                    (j == 0 && i >= 6) ||
                    (j == 1 && i >= 7) ||
                    (j == 2 && i >= 8) ||
                    (j == 3 && i >= 9) ||
                    (j == 4 && i >= 10) ||
                    (j == 6 && i <= 0) ||
                    (j == 7 && i <= 1) ||
                    (j == 8 && i <= 2) ||
                    (j == 9 && i <= 3) ||
                    (j == 10 && i <= 4) 

                )
                {
                    continue;
                }
                
                var roadX = 0.433 * (hexSize + roadSize) * (i-3) - 0.2165 * (hexSize + roadSize) * (j-1);
                var roadY = 0.375 * (hexSize + roadSize) * (j-5);
                var rotation = 0;
                if (j%2 == 0 && i % 2 == 0) {
                    rotation = -1.04719755
                }else if (j%2 == 0 && i % 2 == 1) {
                    rotation = 1.04719755
                }
                var road = new Road(roadX, roadY, 3*roadSize, 2.4*roadSize, rotation);
                this.addObject(road);
            }
        }

        this.draw();

    }

}

