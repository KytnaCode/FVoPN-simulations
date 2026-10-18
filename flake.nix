{
  description = "Fair Voting on Permissionless Networks Simulations";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    systems = ["x86_64-linux"];
    eachSystem = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs {inherit system;}));
  in {
    devShells = eachSystem (pkgs: {
      default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          python3Packages.numpy
        ];

        buildInputs = with pkgs; [
          (python3.withPackages (p: with p; [jupyterlab]))
          uv
        ];
      };
    });
  };
}
