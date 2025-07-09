

/**
 * EEQAUD Kernel RayTracer
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h> 
#include <windows.h>
#include <stdlib.h>
#include <sstream>
#include <string>


#include <glad/gl.h>
#include <GLFW/glfw3.h>


#define N_MAX 1078
#define BMP_HEADER_SIZE 1078

#define texWidth 64
#define texHeight 64

#define IM_WIDTH 640
#define IM_HEIGHT 480
const double M_PI = 3.14159265358979323846;

void read_obj(const char* filename, float** vertex, float** normals, float** color, int* n_vertex, int** faces, int* n_faces);

int write_bmp(const char* filename, int width, int height, unsigned char* rgb);

void readBMP_RGB(char* filename, unsigned char** data_rgb, int* width_rgb, int* height_rgb);
//void processInput(GLFWwindow* window, double* playerX, double* playerY, double* playerAngle, double fovScale, int* map);
void processInput(GLFWwindow* window, double* playerX, double* playerY, double* playerAngle, double fovScale, int mapWidth, int* map);

// Dimensiones de la ventana


GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

GLFWwindow* initOpenGL();
///////////  GPU
__global__ void rayCasting(unsigned char* framebuffer, int screenWidth, int screenHeight, int mapWidth, int mapHeight,
	int* map, unsigned char* textureAtlas, int atlasWidth, int atlasHeight, int texCellWidth, int texCellHeight,
	double posX, double posY, double playerAngle, double fovScale) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	double dirX = cos(playerAngle);
	double dirY = sin(playerAngle);
	// Para un FOV de ~60 grados, plane es perpendicular a dir y escalado
	double planeX = -sin(playerAngle) * fovScale;
	double planeY = cos(playerAngle) * fovScale;

	if (x >= screenWidth) return;
	// 1. Calcular raycasting para columna 'x'
	double cameraX = 2.0 * x / double(screenWidth) - 1;
	double rayDirX = dirX + planeX * cameraX;
	double rayDirY = dirY + planeY * cameraX;

	// Posiciones del mapa 
	int mapX = int(posX);
	int mapY = int(posY);
	// Longitudes del rayo hasta cruzar un cuadrado
	double deltaDistX = fabs(1 / rayDirX);
	double deltaDistY = fabs(1 / rayDirY);

	double sideDistX, sideDistY;
	int stepX, stepY;
	int hit = 0, side;

	double perpWallDist;

	if (rayDirX < 0) {
		stepX = -1;
		sideDistX = (posX - mapX) * deltaDistX;
	}
	else {
		stepX = 1;
		sideDistX = (mapX + 1.0 - posX) * deltaDistX;
	}
	if (rayDirY < 0) {
		stepY = -1;
		sideDistY = (posY - mapY) * deltaDistY;
	}
	else {
		stepY = 1;
		sideDistY = (mapY + 1.0 - posY) * deltaDistY;
	}
	//ABSOLUTE CINEMA
	// DDA
	while (hit == 0) {
		if (sideDistX < sideDistY) {
			sideDistX += deltaDistX;
			mapX += stepX;
			side = 0;
		}
		else {
			sideDistY += deltaDistY;
			mapY += stepY;
			side = 1;
		}

		// ❗ Corte seguro si se sale del mapa
		if (mapX < 0 || mapX >= mapWidth || mapY < 0 || mapY >= mapHeight) {
			hit = 1;
			perpWallDist = 1e6; // simula "pared muy lejana"
			break;
		}

		if (map[mapY * mapWidth + mapX] > 0) {
			hit = 1;
		}
	}


	if (side == 0)
		perpWallDist = (sideDistX - deltaDistX);
	else
		perpWallDist = (sideDistY - deltaDistY);

	// Altura de la pared a dibujar en pantalla
	int lineHeight = (int)(screenHeight / perpWallDist);
	int drawStart = -lineHeight / 2 + screenHeight / 2;
	int drawEnd = lineHeight / 2 + screenHeight / 2;

	if (drawStart < 0) drawStart = 0;
	if (drawEnd >= screenHeight) drawEnd = screenHeight - 1;

	// Pintar cielo arriba
	for (int y = 0; y < drawStart; y++) {
		int idx = (y * screenWidth + x) * 3;
		framebuffer[idx + 0] = 135; // R cielo
		framebuffer[idx + 1] = 206; // G
		framebuffer[idx + 2] = 235; // B
	}

	// Pintar suelo abajo
	for (int y = drawEnd + 1; y < screenHeight; y++) {
		int idx = (y * screenWidth + x) * 3;
		framebuffer[idx + 0] = 68; // R suelo
		framebuffer[idx + 1] = 68;
		framebuffer[idx + 2] = 68;
	}

	// === TEXTURIZACIÓN ===

	// Posición exacta del impacto en la pared
	double wallX;
	if (side == 0)
		wallX = posY + perpWallDist * rayDirY;
	else
		wallX = posX + perpWallDist * rayDirX;

	wallX -= floor(wallX); // solo la parte fraccionaria

	int texX = int(wallX * double(texCellWidth));
	if ((side == 0 && rayDirX > 0) || (side == 1 && rayDirY < 0))
		texX = texCellWidth - texX - 1;

	// Para cada pixel vertical de la pared, calcular color y copiar
	for (int y = drawStart; y <= drawEnd; y++) {
		int d = y * 256 - screenHeight * 128 + lineHeight * 128;
		int texY = ((d * texCellHeight) / lineHeight) / 256;

		int atlasX = mapX * texCellWidth + texX;
		int atlasY = mapY * texCellHeight + texY;

		// Evitar accesos fuera de los límites
		if (atlasX < 0 || atlasX >= atlasWidth || atlasY < 0 || atlasY >= atlasHeight)
			continue;

		int texIdx = (atlasY * atlasWidth + atlasX) * 3;
		int fbIdx = (y * screenWidth + x) * 3;

		framebuffer[fbIdx + 0] = textureAtlas[texIdx + 0];
		framebuffer[fbIdx + 1] = textureAtlas[texIdx + 1];
		framebuffer[fbIdx + 2] = textureAtlas[texIdx + 2];
	}
}

//kernel para procesado del mapa
__global__ void drawMapTextureKernel(unsigned char* framebuffer, int fbWidth, int fbHeight,
	int* map, int mapWidth, int mapHeight,
	unsigned char* textureAtlas, int imgWidth, int imgHeight,
	int texCellWidth, int texCellHeight, int cellPixelSize, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= fbWidth || y >= fbHeight) return;

	int cellX = x / cellPixelSize;
	int cellY = y / cellPixelSize;

	if (cellX >= mapWidth || cellY >= mapHeight) return;

	int pixelInCellX = x % cellPixelSize;
	int pixelInCellY = y % cellPixelSize;

	int cellType = map[cellY * mapWidth + cellX];

	// Textura de la celda dentro del atlas (asumiendo concatenadas horizontalmente)
	int textureOffsetX = cellType * texCellWidth;

	int texX = pixelInCellX * texCellWidth / cellPixelSize;
	int texY = pixelInCellY * texCellHeight / cellPixelSize;

	int imgX = textureOffsetX + texX;
	int imgY = texY;

	int imgIdx = (imgY * imgWidth + imgX) * channels;
	int fbIdx = (y * fbWidth + x) * channels;

	for (int c = 0; c < channels; c++) {
		framebuffer[fbIdx + c] = textureAtlas[imgIdx + c];
	}
}

/*
// crea un buffer y lo registra en cuda --Para despues-- porque hay que instalar mierdas
void createVBO(GLuint* vbo) {
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
}
*/


// Entrada de usuario
double mouseX = 0, mouseY = 0;
bool keys[1024] = { false };

// Callbacks GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key >= 0 && key < 1024) {
		keys[key] = (action != GLFW_RELEASE);
	}
}
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	mouseX = xpos;
	mouseY = ypos;
}

GLFWwindow* initOpenGL() {
	//inicializamos glfw para poder crear ventana
	if (!glfwInit()) {
		std::cerr << "GLFW init failed\n";
		return nullptr;
	}
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	//version 3.3 (la instalada)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Creamos ventana
	GLFWwindow* window = glfwCreateWindow(IM_WIDTH, IM_HEIGHT, "OpenGL + CUDA Interop", nullptr, nullptr);
	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return nullptr;
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	//carga de funciones de openGL modernas
	if (!gladLoadGL(glfwGetProcAddress)) {
		std::cerr << "Error: no se pudo cargar OpenGL con glad\n";
		return nullptr;
	}

	glViewport(0, 0, IM_WIDTH, IM_HEIGHT);
	glPointSize(1.0f); // Tamaño del punto

	return window;
}

__global__ void buildMapFromImage(unsigned char* image, int imgWidth, int imgHeight, int cellSize, int* mapOut) {
	int mapX = blockIdx.x * blockDim.x + threadIdx.x;
	int mapY = blockIdx.y * blockDim.y + threadIdx.y;

	if (mapX >= imgWidth / cellSize || mapY >= imgHeight / cellSize) return;

	int startX = mapX * cellSize;
	int startY = mapY * cellSize;

	// Leer pixel central de la celda
	int pixelIndex = ((startY + cellSize / 2) * imgWidth + (startX + cellSize / 2)) * 3;
	unsigned char r = image[pixelIndex + 0];
	unsigned char g = image[pixelIndex + 1];
	unsigned char b = image[pixelIndex + 2];

	int value;
	double luminosidad = 0.299 * r + 0.587 * g + 0.114 * b;
	// Traduces color a tipo de celda (ejemplo simple)
	if (luminosidad > 200) {
		value = 0; // suelo
	}
	else {
		value = 1; // pared
	}

	mapOut[mapY * (imgWidth / cellSize) + mapX] = value;
}


double moveSpeed = 0.05;

/*
#define mapWidth  24
#define mapHeight 24
*/
void processInput(GLFWwindow* window, double* playerX, double* playerY, double* playerAngle, double fovScale, int mapWidth, int* map) {
	static double lastMouseX;
	static bool firstMouse = true;

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (firstMouse) {
		lastMouseX = xpos;
		firstMouse = false;
	}

	double mouseDeltaX = xpos - lastMouseX;
	lastMouseX = xpos;

	// Sensibilidad del ratón
	double mouseSensitivity = 0.003;
	*playerAngle += mouseDeltaX * mouseSensitivity;

	// Normalizar ángulo entre 0 y 2PI
	if (*playerAngle < 0) *playerAngle += 2 * M_PI;
	if (*playerAngle >= 2 * M_PI) *playerAngle -= 2 * M_PI;

	// Calcular nueva dirección en base al ángulo actualizado
	double dirX = cos(*playerAngle);
	double dirY = sin(*playerAngle);
	double planeX = -sin(*playerAngle) * fovScale;
	double planeY = cos(*playerAngle) * fovScale;

	double newX = *playerX;
	double newY = *playerY;

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		std::cout << "W PRESSED\n";
		newX += dirX * moveSpeed;
		newY += dirY * moveSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		newX -= dirX * moveSpeed;
		newY -= dirY * moveSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		newX -= planeX * moveSpeed;
		newY -= planeY * moveSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		newX += planeX * moveSpeed;
		newY += planeY * moveSpeed;
	}

	// Verificación de colisión
	int mapIndexX = int(newX);
	int mapIndexY = int(*playerY);
	if (map[mapIndexY * mapWidth + mapIndexX] == 0) {
		*playerX = newX;
	}

	mapIndexX = int(*playerX);
	mapIndexY = int(newY);
	if (map[mapIndexY * mapWidth + mapIndexX] == 0) {
		*playerY = newY;
	}
}


/////// Main

int main()
{
	int ImageWidth = 640;
	int ImageHeight = 480;

	//lectura imagen mapa
	char* filemap = "./mapa.bmp";
	unsigned char* mapa_rgb;
	unsigned char* d_mapa_rgb;
	int w_rgb;
	int h_rgb;
	readBMP_RGB(filemap, &mapa_rgb, &w_rgb, &h_rgb);
	cudaMalloc(&d_mapa_rgb, w_rgb * h_rgb * 3);
	cudaMemcpy(d_mapa_rgb, mapa_rgb, w_rgb * h_rgb * 3, cudaMemcpyHostToDevice);

	int cellSize = 10;
	int mapWidth = w_rgb / cellSize;   // = 24
	int mapHeight = h_rgb / cellSize; // = 24
	int* d_map;

	/*int map[mapWidth * mapHeight] = {
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,1,
		1,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,3,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	};
	*/
	cudaMalloc(&d_map, mapWidth * mapHeight * sizeof(int));

	//ventana
	GLFWwindow* window = initOpenGL();
	if (!window) return -1;
	//cargamos texturas
	glViewport(0, 0, ImageWidth, ImageHeight);
	GLuint texID;
	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, ImageWidth, ImageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	//


	// Posición y dirección del jugador
	double playerX = 6.25;
	double playerY = 6.5;
	double fovScale = 0.66;
	/*double dirX = -1.0;
	double dirY = 0.0;
	double planeX = 0.0;
	double planeY = 0.66;
	*/

	dim3 blockDim1(16, 16);
	dim3 gridDim1((mapWidth + blockDim1.x - 1) / blockDim1.x, (mapHeight + blockDim1.y - 1) / blockDim1.y);

	//esto retorna en d_map un array con los valores y si hay pared o no etc en funcion de la imagen, 
	// o sea es el equivalente a hacer unnmemcpy de map
	buildMapFromImage << <gridDim1, blockDim1 >> > (d_mapa_rgb, w_rgb, h_rgb, cellSize, d_map);
	cudaDeviceSynchronize();
	int* h_map = new int[mapWidth * mapHeight];
	cudaMemcpy(h_map, d_map, mapWidth * mapHeight * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "MAPA GENERADO: perro\n";
	for (int y = 0; y < mapHeight; y++) {
		for (int x = 0; x < mapWidth; x++) {
			std::cout << h_map[y * mapWidth + x] << " ";
		}
		std::cout << "\n";
	}
	// Framebuffer
	size_t fbSize = ImageWidth * ImageHeight * 3; //sizeof(unsigned char);
	unsigned char* framebuffer = new unsigned char[ImageWidth * ImageHeight * 3];

	// GPU buffers
	unsigned char* d_framebuffer;
	double playerAngle = 45;
	cudaMalloc(&d_framebuffer, fbSize);
	/*int* d_map;
	cudaMalloc(&d_map, mapWidth * mapHeight * sizeof(int));
	cudaMemcpy(d_map, map, mapWidth * mapHeight * sizeof(int), cudaMemcpyHostToDevice);
	*/

	dim3 blockDim(32);
	dim3 gridDim((ImageWidth + blockDim.x - 1) / blockDim.x);

	//bucle de procesado mientras no se cierre la ventana no para
	while (!glfwWindowShouldClose(window)) {

		double dirX = cos(playerAngle);
		double dirY = sin(playerAngle);
		double planeX = -sin(playerAngle) * fovScale;
		double planeY = cos(playerAngle) * fovScale;
		processInput(window, &playerX, &playerY, &playerAngle, fovScale, mapWidth, h_map);
		//copiamos mapa 


		// Configurar kernel


		rayCasting << <gridDim, blockDim >> > (d_framebuffer, ImageWidth, ImageHeight, mapWidth, mapHeight, d_map, d_mapa_rgb, w_rgb, h_rgb, cellSize, cellSize, playerX, playerY, playerAngle, fovScale);//dirX, dirY, planeX, planeY);

		cudaDeviceSynchronize();

		cudaMemcpy(framebuffer, d_framebuffer, fbSize, cudaMemcpyDeviceToHost);
		//le decimos a openGL que use la textura que tiene textID
		glBindTexture(GL_TEXTURE_2D, texID);
		//copiamos buffer de la cpu a la textura activa en la ventana en GPU
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ImageWidth, ImageHeight, GL_RGB, GL_UNSIGNED_BYTE, framebuffer);
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			std::cerr << "OpenGL error: " << err << std::endl;
		}

		//la desactivamos para no modificarla despues sin querer
		glBindTexture(GL_TEXTURE_2D, 0);

		//limpia ventana
		glClear(GL_COLOR_BUFFER_BIT);
		//activamos texturas de openGL y que vamos a usar la textura textID para dibujar 
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texID);
		//dibujamos un cuadrado que ocupa toda la pantalla y cada vertuce indica la posicion entre -1
		// y 1 que es toda la ventana, cada coordenada indica que parte de la imagen va en ese vertice
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex2f(-1, -1);
		glTexCoord2f(1, 0); glVertex2f(1, -1);
		glTexCoord2f(1, 1); glVertex2f(1, 1);
		glTexCoord2f(0, 1); glVertex2f(-1, 1);
		glEnd();
		//desactivamos textura y texturizado para no afectar a otras partes
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);

		// cambiamos buffer actual por el actualizado y dejamos que procese eventos para que siga funcionando
		glfwSwapBuffers(window);
		glfwPollEvents();


	}
	// Guardar imagen
	/*
	unsigned char* rgb = new unsigned char[ImageWidth * ImageHeight * 3];
	for (int i = 0; i < ImageWidth * ImageHeight; ++i) {
		uint32_t color = framebuffer[i];
		rgb[i * 3 + 0] = (color >> 16) & 0xFF; // R
		rgb[i * 3 + 1] = (color >> 8) & 0xFF;  // G
		rgb[i * 3 + 2] = color & 0xFF;         // B
	}

	// Guardar como BMP
	if (!write_bmp("output.bmp", ImageWidth, ImageHeight, rgb)) {
		std::cerr << "Error escribiendo BMP\n";
	}

	delete[] rgb;
	*/
	cudaFree(d_framebuffer);
	cudaFree(d_map);
	cudaFree(d_mapa_rgb);
	delete[] mapa_rgb;
	delete[] framebuffer;

	std::cout << "Imagen renderizada en output.ppm\n";
	return 0;

}

//////// read/write files

struct BMPHeader
{
	char bfType[2];       /* "BM" */
	int bfSize;           /* Size of file in bytes */
	int bfReserved;       /* set to 0 */
	int bfOffBits;        /* Byte offset to actual bitmap data (= 54) */
	int biSize;           /* Size of BITMAPINFOHEADER, in bytes (= 40) */
	int biWidth;          /* Width of image, in pixels */
	int biHeight;         /* Height of images, in pixels */
	short biPlanes;       /* Number of planes in target device (set to 1) */
	short biBitCount;     /* Bits per pixel (24 in this case) */
	int biCompression;    /* Type of compression (0 if no compression) */
	int biSizeImage;      /* Image size, in bytes (0 if no compression) */
	int biXPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biYPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biClrUsed;        /* Number of colors in the color table (if 0, use
						  maximum allowed by biBitCount) */
	int biClrImportant;   /* Number of important colors.  If 0, all colors
						  are important */
};

void read_obj(const char* filename, float** vertex, float** normals, float** color, int* n_vertex, int** faces, int* n_faces)
{
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		printf("Impossible to open the file !\n");
		return;
	}

	int n_faces_l = 0;
	int faces_l[N_MAX * 3];
	int n_vertex_l = 0;
	float vertex_l[N_MAX * 3];
	float colors_l[N_MAX * 3];
	int n_normals_l = 0;
	float normals_l[N_MAX * 3];

	while (1) {

		char lineHeader[128];
		// Lee la primera palabra de la línea
		int res = fscanf(file, "%s", lineHeader);

		if (res == EOF)
			break; // EOF = End Of File, es decir, el final del archivo. Se finaliza el ciclo.

		if (strcmp(lineHeader, "v") == 0) {
			float x, y, z, r, g, b;
			fscanf(file, "%f %f %f %f %f %f \n", &x, &y, &z, &r, &g, &b);

			// guardo vertices 
			vertex_l[n_vertex_l * 3] = x;
			vertex_l[n_vertex_l * 3 + 1] = y;
			vertex_l[n_vertex_l * 3 + 2] = z;

			// guardo colores
			colors_l[n_vertex_l * 3] = r;
			colors_l[n_vertex_l * 3 + 1] = g;
			colors_l[n_vertex_l * 3 + 2] = b;

			n_vertex_l++;
		}

		if (strcmp(lineHeader, "vn") == 0) {
			float x, y, z;
			fscanf(file, "%f %f %f\n", &x, &y, &z);

			// guardo normales 
			normals_l[n_normals_l * 3] = x;
			normals_l[n_normals_l * 3 + 1] = y;
			normals_l[n_normals_l * 3 + 2] = z;

			n_normals_l++;
		}

		if (strcmp(lineHeader, "f") == 0) {
			int i, j, k;
			fscanf(file, "%d//%d %d//%d %d//%d\n", &i, &i, &j, &j, &k, &k);

			// guardo caras (comenzando por 0)
			faces_l[n_faces_l * 3] = i - 1;
			faces_l[n_faces_l * 3 + 1] = j - 1;
			faces_l[n_faces_l * 3 + 2] = k - 1;

			n_faces_l++;
		}
	}

	if (n_normals_l != n_vertex_l) {
		printf("Different number of vertex and normals!!!\n");
		return;
	}

	/// copio datos

	n_vertex[0] = n_vertex_l;
	*vertex = new float[n_vertex_l * 3];
	*normals = new float[n_vertex_l * 3];
	*color = new float[n_vertex_l * 3];

	for (int i_v = 0; i_v < n_vertex_l; i_v++) {
		(*vertex)[i_v * 3] = vertex_l[i_v * 3];
		(*vertex)[i_v * 3 + 1] = vertex_l[i_v * 3 + 1];
		(*vertex)[i_v * 3 + 2] = vertex_l[i_v * 3 + 2];

		(*normals)[i_v * 3] = normals_l[i_v * 3];
		(*normals)[i_v * 3 + 1] = normals_l[i_v * 3 + 1];
		(*normals)[i_v * 3 + 2] = normals_l[i_v * 3 + 2];

		(*color)[i_v * 3] = colors_l[i_v * 3];
		(*color)[i_v * 3 + 1] = colors_l[i_v * 3 + 1];
		(*color)[i_v * 3 + 2] = colors_l[i_v * 3 + 2];
	}

	n_faces[0] = n_faces_l;
	*faces = new int[n_faces_l * 3];

	for (int i_f = 0; i_f < n_faces_l; i_f++) {
		(*faces)[i_f * 3] = faces_l[i_f * 3];
		(*faces)[i_f * 3 + 1] = faces_l[i_f * 3 + 1];
		(*faces)[i_f * 3 + 2] = faces_l[i_f * 3 + 2];
	}
}

int write_bmp(const char* filename, int width, int height, unsigned char* rgb)
{
	int i, j, ipos;
	int bytesPerLine;
	char* line;

	FILE* file;
	struct BMPHeader bmph;

	/* The length of each line must be a multiple of 4 bytes */

	bytesPerLine = (3 * (width + 1) / 4) * 4;

	strcpy(bmph.bfType, "BM");
	bmph.bfOffBits = 54;
	bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
	bmph.bfReserved = 0;
	bmph.biSize = 40;
	bmph.biWidth = width;
	bmph.biHeight = height;
	bmph.biPlanes = 1;
	bmph.biBitCount = 24;
	bmph.biCompression = 0;
	bmph.biSizeImage = bytesPerLine * height;
	bmph.biXPelsPerMeter = 0;
	bmph.biYPelsPerMeter = 0;
	bmph.biClrUsed = 0;
	bmph.biClrImportant = 0;

	file = fopen(filename, "wb");
	if (file == NULL) return(0);

	fwrite(&bmph.bfType, 2, 1, file);
	fwrite(&bmph.bfSize, 4, 1, file);
	fwrite(&bmph.bfReserved, 4, 1, file);
	fwrite(&bmph.bfOffBits, 4, 1, file);
	fwrite(&bmph.biSize, 4, 1, file);
	fwrite(&bmph.biWidth, 4, 1, file);
	fwrite(&bmph.biHeight, 4, 1, file);
	fwrite(&bmph.biPlanes, 2, 1, file);
	fwrite(&bmph.biBitCount, 2, 1, file);
	fwrite(&bmph.biCompression, 4, 1, file);
	fwrite(&bmph.biSizeImage, 4, 1, file);
	fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biClrUsed, 4, 1, file);
	fwrite(&bmph.biClrImportant, 4, 1, file);

	line = (char*)malloc(bytesPerLine);

	for (i = height - 1; i >= 0; i--)
	{
		for (j = 0; j < width; j++)
		{
			ipos = 3 * (width * i + j);
			line[3 * j] = rgb[ipos + 2];
			line[3 * j + 1] = rgb[ipos + 1];
			line[3 * j + 2] = rgb[ipos];
		}
		fwrite(line, bytesPerLine, 1, file);
	}

	free(line);
	fclose(file);

	return(1);
}

void readBMP_RGB(char* filename, unsigned char** data_rgb, int* w, int* h)
{
	FILE* f = fopen(filename, "rb");
	char info[54];
	fread(info, sizeof(char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;

	// allocate 3 bytes per pixel
	*data_rgb = new unsigned char[size];

	// read the rest of the data at once
	fread(*data_rgb, sizeof(unsigned char), size, f);

	// close file
	fclose(f);

	// invert some data
	for (int i = 0; i < size; i += 3)
	{
		unsigned char tmp;
		tmp = (*data_rgb)[i];
		(*data_rgb)[i] = (*data_rgb)[i + 2];
		(*data_rgb)[i + 2] = tmp;
	}

	w[0] = width;
	h[0] = height;
}
