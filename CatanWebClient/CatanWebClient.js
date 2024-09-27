class backgroundWater extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);
    }

    draw(ctx, objectCenter, objectSize) {
        // draw images/background_map.png
        var img = new Image();
        img.src = "images/water.png";
        img.onload = () => {
            ctx.drawImage(img, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            this.drawChildren(ctx, objectCenter, objectSize);
        }
    }
}

class backgroundMap extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);
    }

    draw(ctx, objectCenter, objectSize) {
        // draw images/background_map.png
        var img = new Image();
        img.src = "images/background_map.png";
        img.onload = () => {
            ctx.drawImage(img, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            this.drawChildren(ctx, objectCenter, objectSize);
        }
    }
}

class Hex extends CanvasObject {
    constructor(px, py, sx, sy, hexX, hexY) {
        super(px, py, sx, sy);
        this.hexCoords = {x: hexX, y: hexY};
        this.addChild(
            new TextObject(
                0, 0, 
                0.866, 0.15, 
                hexX + ", " + hexY, 
                "Arial", "#000000"
            )
        );
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
        this.player = null;
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
        img.src = "images/road_silver.svg";
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
        var background = new backgroundWater(0, 0, 1, 1);
        this.addObject(background);
        var map = new backgroundMap(0, 0, 1, 1);
        background.addChild(map);

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
                var hex = new Hex(HexX, HexY, hexSize, hexSize, i, j);
                map.addChild(hex);
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
                map.addChild(corner);
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
                map.addChild(road);
            }
        }

        this.draw();

    }

}

